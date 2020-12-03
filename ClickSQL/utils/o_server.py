# coding=utf-8
from multiprocessing import Process
from threading import Thread
from socket import *
import select


# sendfile 零拷贝
def recv_data(new_socket, client_info):
    print("客户端{}已经连接".format(client_info))
    # 接受数据
    raw_data = new_socket.recv(1024)
    while raw_data:
        print(f"收到来自{client_info}的数据：{raw_data}")
        raw_data = new_socket.recv(1024)
    new_socket.close()


# 多进程并发阻塞
def server_process():
    # 实例化socket对象
    socket_server = socket(AF_INET, SOCK_STREAM)
    # 设置端口复用
    socket_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    # 绑定IP地址和端口
    socket_server.bind(("", 7788))
    # 改主动为被动，监听客户端
    socket_server.listen(5)
    while True:
        # 等待连接
        new_socket, client_info = socket_server.accept()
        p = Process(target=recv_data, args=(new_socket, client_info))
        p.start()
        # 多进程会复制父进程的内存空间，所以父进程中new_socket也必须关闭
        new_socket.close()


# 多线程并发阻塞
def server_thread():
    # 实例化socket对象
    socket_server = socket(AF_INET, SOCK_STREAM)
    # 设置端口复用
    socket_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    # 绑定IP地址和端口
    socket_server.bind(("", 7788))
    # 改主动为被动，监听客户端
    socket_server.listen(5)
    while True:
        # 等待连接
        new_socket, client_info = socket_server.accept()
        p = Thread(target=recv_data, args=(new_socket, client_info))
        p.start()
        # 多线程共享一片内存区域，所以这里不用关闭
        # new_socket.close()


# 多路复用IO---select模型
def server_select():
    # 实例化对象
    socket_server = socket(AF_INET, SOCK_STREAM)
    # 绑定IP和端口
    socket_server.bind(("", 7788))
    # 将主动模式改为被动模式
    socket_server.listen(5)
    # 创建套接字列表
    socket_lists = [socket_server]
    # 等待客户端连接
    while True:
        # 只监听读的状态，程序阻塞在这，不消耗CPU，如果列表里面的值读状态变化后，就解阻塞
        read_lists, _, _ = select.select(socket_lists, [], [])
        # 循环有变化的套接字
        for sock in read_lists:
            # 判断是否是主套接字
            if sock == socket_server:
                # 获取新连接
                new_socket, client_info = socket_server.accept()
                print(f"客户端：{client_info}已连接")
                # 添加到监听列表中
                socket_lists.append(new_socket)
            else:
                # 不是主客户端，即接收消息
                raw_data = sock.recv(1024)
                if raw_data:
                    print(f"接收数据：{raw_data.decode('gb2312')}")
                else:
                    # 如果没有数据，则客户端断开连接
                    sock.close()
                    # 从监听列表中删除该套接字
                    socket_lists.remove(sock)


# 多路复用IO---epoll模型 参考https://blog.csdn.net/pysense/article/details/103840680
# 该代码无法在windows上运行，因为epoll是Linux2.6内核增加的新功能，windows并不支持。
def server_epoll():
    # 创建socket对象
    sock_server = socket(AF_INET, SOCK_STREAM)
    # 绑定IP和端口
    sock_server.bind(("", 7788))
    # 将主动模式设置为被动模式，监听连接
    sock_server.listen(5)
    # 创建epoll监测对象（Only supported on Linux 2.5.44 and newer.）
    epoll = select.epoll()
    # print("未注册epoll对象：{}".format(epoll))
    # 注册主套接字,监控读状态
    epoll.register(sock_server.fileno(), select.EPOLLIN)
    # print("注册了主套接字后：{}".format(epoll))
    # 创建字典，保存套接字对象
    sock_dicts = {}
    # 创建字典，保存客户端信息
    client_dicts = {}
    while True:
        # print("所有套接字：{}".format(sock_dicts))
        # print("所有客户端信息：{}".format(client_dicts))
        # 程序阻塞在这，返回文件描述符有变化的对象
        poll_list = epoll.poll()
        # print("有变化的套接字：{}".format(poll_list))
        for sock_fileno, events in poll_list:
            # print("文件描述符：{}，事件：{}".format(sock_fileno, events))
            # 判断是否是主套接字
            if sock_fileno == sock_server.fileno():
                # 创建新套接字
                new_sock, client_info = sock_server.accept()
                print(f"客户端：{client_info}已连接")
                # 注册到epoll监测中
                epoll.register(new_sock.fileno(), select.EPOLLIN)
                # 添加到套接字字典当中
                sock_dicts[new_sock.fileno()] = new_sock
                client_dicts[new_sock.fileno()] = client_info
            else:
                # 接收消息
                raw_data = sock_dicts[sock_fileno].recv(1024)
                if raw_data:
                    print(f"来自{client_dicts[sock_fileno]}的数据：{raw_data.decode('gb2312')}")
                else:
                    # 关闭连接
                    sock_dicts[sock_fileno].close()
                    # 注销epoll监测对象
                    epoll.unregister(sock_fileno)
                    # 数据为空，则客户端断开连接，删除相关数据
                    del sock_dicts[sock_fileno]
                    del client_dicts[sock_fileno]


if __name__ == '__main__':
    server_process()
