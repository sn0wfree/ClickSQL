# coding=utf-8
def chunk(obj, chunks=2000):
    if hasattr(obj, '__len__'):
        length = len(obj)
        for i in range(0, length, chunks):
            yield obj[i:i + chunks]


if __name__ == '__main__':
    pass
