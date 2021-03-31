# coding=utf-8

from ClickSQL.nodes.base import BaseSingleFactorBaseNode
from collections import namedtuple
import pandas as pd

CIK = namedtuple('CoreIndexKeys', ('dts', 'iid'))
FactorInfo = namedtuple('FactorInfo', ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions'))


def SmartDataFrame(df: pd.DataFrame, db_table: str, dts: str, iid: str, origin_factor_names: str, alias: str, sql: str,
                   conditions: str):
    sproperty = {'_db_table': property(lambda x: db_table),
                 '_cik_dt': property(lambda x: dts),
                 '_cik_iid': property(lambda x: iid),
                 '_origin_factor_names': property(lambda x: origin_factor_names),
                 '_alias': property(lambda x: alias),
                 '_sql': property(lambda x: sql),
                 '_conditions': property(lambda x: conditions),

                 }

    result_cls = type('SmartDataFrame', (pd.DataFrame,), sproperty)
    return result_cls(df)


class FactorCheckHelper(object):
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

    @staticmethod
    def check_cik_dt(cik_dt, default_cik_dt):
        if cik_dt is not None:
            pass

        elif cik_dt is None:
            cik_dt = default_cik_dt
        else:
            raise NotImplementedError('cik_dt is not setup!')
        return cik_dt

    @staticmethod
    def check_cik_iid(cik_iid, default_cik_iid):
        if cik_iid is not None:
            pass

        elif cik_iid is None:
            cik_iid = default_cik_iid
        else:
            raise NotImplementedError('cik_dt is not setup!')
        return cik_iid


class FatctorTable(FactorCheckHelper):
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

    # def __check_cik__(self, cik_dt=None, cik_iid=None):
    #     if cik_dt is not None:
    #         pass
    #
    #     elif cik_dt is None:
    #         cik_dt = self._cik.dts
    #     else:
    #         raise NotImplementedError('cik_dt is not setup!')
    #
    #     if cik_iid is not None:
    #         pass
    #     elif cik_iid is None :
    #         cik_iid = self._cik.iid
    #     else:
    #         raise NotImplementedError('cik_iid is not setup!')
    #     return cik_dt, cik_iid

    def setup_cik(self, cik_dt, cik_iid):
        self._cik = CIK(cik_dt, cik_iid)

    # def getDB(self, db):
    #     self.db = db

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
        cik_dt, cik_iid = self.check_cik_dt(cik_dt=cik_dt, default_cik_dt=self._cik.dts), self.check_cik_iid(
            cik_iid=cik_iid, default_cik_iid=self._cik.iid)
        res = self._get_factor_without_check(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid, conds=conds,
                                             as_alias=as_alias)
        self._factors.append(res)

    def show_factors(self, reduced=False, to_df=True):
        if reduced:
            # ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions')
            # ['db_table', 'dts', 'iid', 'conditions']
            cols = list(FactorInfo._fields[:3]) + [FactorInfo._fields[-1]]

            f = pd.DataFrame(self._factors, columns=FactorInfo._fields)
            fgroupby = f.groupby(cols)
            # can_merged_index = (fgroupby['sql'].count() > 1).reset_index()
            # can_merged_index = can_merged_index[can_merged_index['sql']]
            # can_merged_index = fgroupby.count().index
            factors = []
            for (db_table, dts, iid, conditions), df in fgroupby:
                # masks = (f['db_table'] == db_table) & (f['dts'] == dts) & (f['iid'] == iid) & (
                #         f['conditions'] == conditions)
                cc = df[['origin_factor_names', 'alias']].apply(lambda x: ','.join(x))
                origin_factor_names = cc['origin_factor_names'].split(',')
                alias = cc['alias'].split(',')
                origin_factor_names_new, alias_new = zip(*list(set(zip(origin_factor_names, alias))))
                alias_new = list(map(lambda x: x if x != 'None' else None, alias_new))

                cik_dt, cik_iid = self.check_cik_dt(cik_dt=dts, default_cik_dt=self._cik.dts), self.check_cik_iid(
                    cik_iid=iid, default_cik_iid=self._cik.iid)
                res = self._get_factor_without_check(db_table, origin_factor_names_new, cik_dt=cik_dt, cik_iid=cik_iid,
                                                     conds=conditions, as_alias=alias_new)
                factors.append(res)

        else:
            factors = self._factors
        if to_df:
            return pd.DataFrame(factors, columns=FactorInfo._fields)
        else:
            return factors

        # no_duplicates_df = f.eval("+".join(cols))
        ## todo auto merge same condition,dbtable,dts,iid
        # return can_merged_index

    def obtain(self, reduced=True, add_limit=True):
        factors = self.show_factors(reduced=reduced, to_df=False)

        for db_table, dts, iid, origin_factor_names, alias, sql, conditions in factors:
            if add_limit:
                sql = f'select * from ({sql}) limit 100'
            df = self._node(sql)
            res = SmartDataFrame(df, db_table, dts, iid, origin_factor_names, alias, sql, conditions)

            # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
            yield res

    def obtain_merged(self, reduced=True, add_limit=True):
        return pd.concat(
            map(lambda x: x.set_index(['cik_dt', 'cik_iid']), self.obtain(reduced=reduced, add_limit=add_limit)),
            axis=1)

    def where(self, cik_dt: list = None, cik_iid: list = None, cik_dt_format="%Y%m%d"):
        extra_conds = []
        if cik_dt is None:
            pass
        else:
            cik_dt_ = ','.join(pd.to_datetime(cik_dt).strftime(cik_dt_format))
            cik_dt_cond = f"cik_dt in ({cik_dt_}) "

        if cik_iid is None:
            pass


if __name__ == '__main__':
    FatctorTable()
    pass
