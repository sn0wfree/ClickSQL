# coding=utf-8

from ClickSQL.nodes.base import BaseSingleFactorBaseNode
from collections import namedtuple

CIK = namedtuple('CoreIndexKeys', ('dts', 'iid'))
FactorInfo = namedtuple('FactorInfo', ('sql', 'dts', 'iid'))


class FatctorTable(object):
    __Name__ = "基础因子库单因子表"

    def __init__(self, *args, **kwargs):
        # super(FatctorTable, self).__init__(*args, **kwargs)
        self._node = BaseSingleFactorBaseNode(*args, **kwargs)

        cik_dt = None if 'cik_dt' not in kwargs.keys() else kwargs['cik_dt']
        cik_iid = None if 'cik_iid' not in kwargs.keys() else kwargs['cik_iid']
        self._cik = CIK(cik_dt, cik_iid)
        self._checked = False
        self._factors = []

    def __auto_check_cik__(self):
        if not self._checked and (self._cik.dts is None or self._cik.iid is None):
            raise NotImplementedError('cik(dts or iid) is not setup!')
        else:
            self._checked = True

    def __check_cik__(self, cik_dt=None, cik_iid=None):
        if cik_dt is None and self._checked:
            cik_dt = self._cik.dts
        else:
            raise NotImplementedError('cik_dt is not setup!')
        if cik_iid is None and self._checked:
            cik_iid = self._cik.iid
        else:
            raise NotImplementedError('cik_iid is not setup!')
        return cik_dt, cik_iid

    def setup_cik(self, cik_dt, cik_iid):
        self._cik = CIK(cik_dt, cik_iid)

    # def getDB(self, db):
    #     self.db = db
    def _getFactor(self, db_table, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None):
        cik_dt, cik_iid = self.__check_cik__(cik_dt=cik_dt, cik_iid=cik_iid)

        if isinstance(factor_names, str):
            factor_names = [factor_names]
        elif isinstance(factor_names, (list, tuple, set)):
            factor_names = list(factor_names)
        else:
            raise ValueError('columns only accept list tuple str!')

        cols_str = ','.join(factor_names + [cik_dt, cik_iid])
        sql = f'select {cols_str} from {db_table}'

        return FactorInfo(sql, cik_dt, cik_iid)


if __name__ == '__main__':
    pass
