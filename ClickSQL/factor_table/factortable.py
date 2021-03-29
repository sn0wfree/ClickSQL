# coding=utf-8

from ClickSQL.nodes.base import BaseSingleFactorBaseNode
from collections import namedtuple
import pandas as pd

CIK = namedtuple('CoreIndexKeys', ('dts', 'iid'))
FactorInfo = namedtuple('FactorInfo', ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions'))


class FatctorTable(object):
    __Name__ = "基础因子库单因子表"

    def __init__(self, *args, **kwargs):
        # super(FatctorTable, self).__init__(*args, **kwargs)
        self._node = BaseSingleFactorBaseNode(*args, **kwargs)

        cik_dt = None if 'cik_dt' not in kwargs.keys() else kwargs['cik_dt']
        cik_iid = None if 'cik_iid' not in kwargs.keys() else kwargs['cik_iid']
        self._cik = CIK(cik_dt, cik_iid)
        self._checked = False
        self.__auto_check_cik__()
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
    @staticmethod
    def _check_alias(factor_names: (list,), as_alias: (list, tuple, str) = None):
        if as_alias is None:
            alias = len(factor_names) * [None]
        elif isinstance(as_alias, str):
            alias = [as_alias]
        elif isinstance(as_alias, (list, tuple)):
            if len(as_alias) != len(factor_names):
                raise ValueError('as_alias is not match factor_names')
            else:
                alias = as_alias
        else:
            raise ValueError('alias only accept list tuple str!')
        return alias

    @staticmethod
    def _check_factor_names(factor_names: (list, tuple, str)):
        if isinstance(factor_names, str):
            factor_names = [factor_names]
        elif isinstance(factor_names, (list, tuple)):
            factor_names = list(factor_names)
        else:
            raise ValueError('columns only accept list tuple str!')
        return factor_names

    @classmethod
    def _get_factor_without_check(cls, db_table, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
                                  conds: str = '1', as_alias: (list, tuple, str) = None):
        """

        :param db_table:
        :param factor_names:
        :param cik_dt:
        :param cik_iid:
        :param conds:  conds = @test1>1 | @test2<1
        :return:
        """
        factor_names = cls._check_factor_names(factor_names)

        alias = cls._check_alias(factor_names, as_alias=as_alias)
        # rename variables
        f_names_list = [f if (a is None) or (f == a) else f"{f} as {a}" for f, a in zip(factor_names, alias)]

        cols_str = ','.join(f_names_list)

        conditions = '1' if conds == '1' else conds.replace('&', 'and').replace('|', 'or').replace('@', '')
        cik_dt_str = f"{cik_dt} as cik_dt" if cik_dt != 'cik_dt' else cik_dt
        cik_iid_str = f"{cik_iid} as cik_iid" if cik_iid != 'cik_iid' else cik_iid

        sql = f'select {cols_str}, {cik_dt_str}, {cik_iid_str}  from {db_table} where {conditions}'

        return FactorInfo(db_table, cik_dt, cik_iid, ','.join(map(str, factor_names)), ','.join(map(str, alias)),
                          sql, conds)  #

    def add_factor(self, db_table, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None, conds: str = '1',
                   as_alias: (list, tuple, str) = None):
        cik_dt, cik_iid = self.__check_cik__(cik_dt=cik_dt, cik_iid=cik_iid)
        res = self._get_factor_without_check(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid, conds=conds,
                                             as_alias=as_alias)
        self._factors.append(res)

    def show_factors(self):
        return pd.DataFrame(self._factors, columns=FactorInfo._fields)

    def __auto_merge__(self):
        cols = ['db_table', 'dts', 'iid', 'conditions']
        f = self.show_factors()
        can_merged_index = f.groupby(cols).count() > 1
        no_duplicates_df = f.eval( "+".join(cols))
        ## todo auto merge same condition,dbtable,dts,iid
        return can_merged_index


if __name__ == '__main__':
    FatctorTable()
    pass
