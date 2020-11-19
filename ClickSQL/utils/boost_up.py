# coding=utf-8

def boost_up(func, tasks, core=None, method='Process', chunksize=None, star=True):
    if method == 'Process':
        from multiprocessing import Pool

    else:
        from multiprocessing.dummy import Pool

    pool = Pool(core)
    if star:
        h = pool.starmap(func, tasks, chunksize=chunksize)
    else:
        h = pool.map(func, tasks, chunksize=chunksize)
    pool.close()
    pool.join()
    return h


if __name__ == '__main__':
    pass
