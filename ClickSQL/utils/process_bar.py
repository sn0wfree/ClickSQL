# coding=utf-8
import datetime
import sys
import time


def progress_test(counts, lenfile, speed):
    bar_length = 20
    w = (lenfile - counts) * speed
    eta = time.time() + w
    precent = counts / float(lenfile)

    ETA = datetime.datetime.fromtimestamp(eta)
    hashes = '#' * int(precent * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("""\r%d%%|%s|read %d projects|Speed : %.4f |ETA: %s """ % (
        precent * 100, hashes + spaces, counts, speed, ETA))

    # sys.stdout.write("\rthis spider has already read %d projects, speed: %.4f/projects" % (counts,f2-f1))

    # sys.stdout.write("\rPercent: [%s] %d%%,remaining time: %.4f mins"%(hashes + spaces,precent,w))
    sys.stdout.flush()


def process_bar(iterable_obj, counts=None):
    if hasattr(iterable_obj, '__len__'):
        counts = len(iterable_obj)
    elif isinstance(counts, int):
        pass  # counts = counts
    else:
        iterable_obj = list(iterable_obj)
        counts = len(iterable_obj)
    for count, i in enumerate(iterable_obj):
        f = time.time()
        yield i

        progress_test(count, counts, time.time() - f)


if __name__ == '__main__':
    pass
