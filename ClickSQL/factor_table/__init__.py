# coding=utf-8

from collections import namedtuple, deque, Callable

import pandas as pd

from ClickSQL.nodes.base import BaseSingleQueryBaseNode

CIK = namedtuple('CoreIndexKeys', ('dts', 'iid'))
CIKDATA = namedtuple('CoreIndexKeys', ('dts', 'iid'))
FactorInfo = namedtuple('FactorInfo',
                        ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'via', 'conditions'))


# def SmartDataFrame(df: pd.DataFrame, db_table: str, dts: str, iid: str, origin_factor_names: str, alias: str, sql: str,
#                    via: str,
#                    conditions: str):
#     sproperty = {'_db_table': property(lambda x: db_table),
#                  '_cik_dt': property(lambda x: dts),
#                  '_cik_iid': property(lambda x: iid),
#                  '_origin_factor_names': property(lambda x: origin_factor_names),
#                  '_alias': property(lambda x: alias),
#                  '_sql': property(lambda x: sql),
#                  '_conditions': property(lambda x: conditions),
#                  '_via': property(lambda x: via),
#
#                  }
#
#     result_cls = type('SmartDataFrame', (pd.DataFrame,), sproperty)
#     return result_cls(df)


class FactorCheckHelper(object):
    @staticmethod
    def generate_alias(factor_names: (list,), as_alias: (list, tuple, str) = None):
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
    def generate_factor_names(factor_names: (list, tuple, str)):
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


class __MetaFactorTable__(FactorCheckHelper):
    pass


class _Factors(deque):
    @classmethod
    def add_factor(cls, db_table: str, factor_names: (list, tuple, str), cik_dt,
                   cik_iid, cik_dt_format: str = 'datetime', as_alias: (list, tuple, str) = None,
                   conds='1'):
        if isinstance(db_table, str):
            if db_table.lower().startswith('select'):
                via = 'sql'
                return cls.add_factor_via_sql(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
                                              as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format)
            else:
                via = 'db_table'
                return cls.add_factor_via_db_table(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
                                                   as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format)
        elif isinstance(db_table, pd.DataFrame):

            cls.add_factor_via_df(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
                                  as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format)

        elif isinstance(db_table, __MetaFactorTable__):
            return list(cls.add_factor_via_ft(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
                                              as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format))
        else:
            raise NotImplementedError('not supported')

    @staticmethod
    def _add_factor_via_obj(db_table: str, factor_names: (list, tuple, str), via: str, cik_dt,
                            cik_iid, cik_dt_format: str = 'datetime', as_alias: (list, tuple, str) = None,
                            conds='1', check=False):
        """

       :param as_alias:
       :param db_table:
       :param factor_names:
       :param cik_dt:
       :param cik_iid:
       :param conds:  conds = @test1>1 | @test2<1
       :return:
       """
        factor_names = FactorCheckHelper.generate_factor_names(factor_names)
        alias = FactorCheckHelper.generate_alias(factor_names, as_alias=as_alias)
        # rename variables
        f_names_list = [f if (a is None) or (f == a) else f"{f} as {a}" for f, a in zip(factor_names, alias)]
        cols_str = ','.join(f_names_list)

        # change dt dtype for suitable dtype
        conditions = '1' if conds == '1' else conds.replace('&', 'and').replace('|', 'or').replace('@', '')

        # convert cik_dt
        if cik_dt == 'cik_dt':
            cik_dt_str = cik_dt
        else:
            if cik_dt_format == 'str':
                cik_dt_str = f"parseDateTimeBestEffort({cik_dt}) as cik_dt"
            elif cik_dt_format == 'datetime':
                cik_dt_str = f"{cik_dt} as cik_dt"
            elif cik_dt_format == 'int':
                cik_dt_str = f"parseDateTimeBestEffort(toString({cik_dt})) as cik_dt"
            else:
                cik_dt_str = f"parseDateTimeBestEffort(toString({cik_dt})) as cik_dt"

        # convert cik_iid
        cik_iid_str = f"{cik_iid} as cik_iid"
        if via == 'sql':
            sql = f'select {cols_str}, {cik_dt_str}, {cik_iid_str}  from ({db_table}) where {conditions}'
        else:
            sql = f'select {cols_str}, {cik_dt_str}, {cik_iid_str}  from {db_table} where {conditions}'

        res = FactorInfo(db_table, cik_dt, cik_iid, ','.join(map(str, factor_names)), ','.join(map(str, alias)),
                         sql, via, conds)  #
        return res
        # self.append(res)

    @classmethod
    def add_factor_via_db_table(cls, db_table: str, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
                                as_alias: (list, tuple, str) = None, conds='1', cik_dt_format: str = 'datetime', ):
        return cls._add_factor_via_obj(db_table, factor_names, 'db_table', cik_dt=cik_dt, cik_dt_format=cik_dt_format,
                                       cik_iid=cik_iid, as_alias=as_alias, conds=conds)

    @classmethod
    def add_factor_via_sql(cls, sql_ori, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
                           cik_dt_format: str = 'datetime',
                           as_alias: (list, tuple, str) = None, conds='1'):
        return cls._add_factor_via_obj(sql_ori, factor_names, 'sql', cik_dt=cik_dt, cik_dt_format=cik_dt_format,
                                       cik_iid=cik_iid, as_alias=as_alias, conds=conds)

    @classmethod
    def add_factor_via_df(cls, *args, **kwargs):
        # todo add factor via dataframe
        raise NotImplementedError('not supported')
        pass

    @classmethod
    def add_factor_via_ft(cls, factor_table: __MetaFactorTable__, factor_names: (list, tuple), cik_dt=None,
                          cik_iid=None,
                          as_alias: (list, tuple, str) = None, conds='1', cik_dt_format: str = 'datetime', ):
        # todo add factor via factortable
        if isinstance(factor_table, __MetaFactorTable__):
            for f in factor_table._factors.show_factors(reduced=True, to_df=False):  # get one factorInfo
                origin_factor_names = f.origin_factor_names.split(',')
                alias_factor_names = f.alias.split(',')
                for o_f, alias_f in zip(origin_factor_names, alias_factor_names):  # mapping factor and its alias
                    if o_f in factor_names or alias_f in factor_names:
                        # if required factor name at factor or alias, will yield this factorinfo and break this loop
                        yield f
                        # todo optimize the add factor via ft process!!
                        # todo if get multifactors, need get required factors only rather than the whole factorinfo
                        break

            # raise NotImplementedError('not supported')
        else:
            raise TypeError('factor_table must be FactorTable or its subclass!')

    def show_factors(self, reduced=False, to_df=True):
        # todo unstack factor name to check whether factor exists duplicates!!
        if reduced:
            # ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql','via', 'conditions')
            # ['db_table', 'dts', 'iid', 'conditions']
            cols = list(FactorInfo._fields[:3]) + list(FactorInfo._fields[-2:])

            f = pd.DataFrame(list(self), columns=FactorInfo._fields)
            factor_name_col = FactorInfo._fields[3]
            alias_col = FactorInfo._fields[4]

            # can_merged_index = (fgroupby['sql'].count() > 1).reset_index()
            # can_merged_index = can_merged_index[can_merged_index['sql']]
            # can_merged_index = fgroupby.count().index
            factors = []
            for (db_table, dts, iid, via, conditions), df in f.groupby(cols):
                # masks = (f['db_table'] == db_table) & (f['dts'] == dts) & (f['iid'] == iid) & (
                #         f['conditions'] == conditions)
                cc = df[[factor_name_col, alias_col]].apply(lambda x: ','.join(x))
                origin_factor_names = cc[factor_name_col].split(',')
                alias = cc[alias_col].split(',')
                origin_factor_names_new, alias_new = zip(*list(set(zip(origin_factor_names, alias))))
                alias_new = list(map(lambda x: x if x != 'None' else None, alias_new))

                # cik_dt, cik_iid = self.check_cik_dt(cik_dt=dts, default_cik_dt=self._cik.dts), self.check_cik_iid(
                #     cik_iid=iid, default_cik_iid=self._cik.iid)
                # add_factor process have checked
                res = self._add_factor_via_obj(db_table, origin_factor_names_new, via, cik_dt=dts, cik_iid=iid,
                                               conds=conditions, as_alias=alias_new)
                factors.append(res)
        else:
            factors = self
        if to_df:
            return pd.DataFrame(factors, columns=FactorInfo._fields)
        else:
            return factors

    def _generate_fetch_sql_iter(self, filter_cond_dts, filter_cond_ids, reduced=True, add_limit=False):
        factors = self.show_factors(reduced=reduced, to_df=False)
        # sql_list = []
        if add_limit:
            limit_str = 'limit 100'
        else:
            limit_str = ''
        ## todo 可能存在性能点
        for db_table, dts, iid, origin_factor_names, alias, sql, via, conditions in factors:
            sql2 = f"select * from ({sql}) where {filter_cond_dts} and {filter_cond_ids} {limit_str} "
            yield sql2

    def fetch_iter(self, query, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False, to_sql=False):
        if not isinstance(query, Callable):
            raise ValueError('query must database connector with __call__')
        sql_list_iter = self._generate_fetch_sql_iter(filter_cond_dts, filter_cond__ids, reduced=reduced,
                                                      add_limit=add_limit)

        if to_sql:
            for sql2 in sql_list_iter:
                yield sql2
        else:

            for sql2 in sql_list_iter:
                df = query(sql2)
                ## remove smartdataframe

                res = pd.DataFrame(df).set_index(['cik_dt', 'cik_iid'])

                # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
                yield res

    def fetch_all(self, query, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False, to_sql=False):
        if not isinstance(query, Callable):
            raise ValueError('query must database connector with __call__')
        sql_list_iter = self._generate_fetch_sql_iter(filter_cond_dts, filter_cond__ids, reduced=reduced,
                                                      add_limit=add_limit)

        from functools import reduce

        def join(sql1, sql2):
            settings = ' settings joined_subquery_requires_alias=0 '
            sql = f"select * from ({sql1}) all full join ({sql2}) using (cik_dt,cik_iid)  {settings}"
            return sql

        s = reduce(lambda x, y: join(x, y), sql_list_iter)
        if to_sql:
            yield s
        else:
            df = query(s)
            res = pd.DataFrame(df).set_index(['cik_dt', 'cik_iid'])

            # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
            yield res


class __FactorTable__(__MetaFactorTable__):
    __Name__ = "基础因子库单因子表"

    def __init__(self, *args, **kwargs):
        # super(FatctorTable, self).__init__(*args, **kwargs)
        self._node = BaseSingleQueryBaseNode(*args, **kwargs)

        cik_dt = None if 'cik_dt' not in kwargs.keys() else kwargs['cik_dt']
        cik_iid = None if 'cik_iid' not in kwargs.keys() else kwargs['cik_iid']
        self._cik = CIK(cik_dt, cik_iid)
        self._cik_data = None
        self._checked = False
        self.__auto_check_cik__()  # check default dt and iid whether set up
        self._factors = _Factors()

        self._strict_cik = False if 'strict_cik' not in kwargs.keys() else kwargs['strict_cik']
        # self.append = self.add_factor

        self._cik_dts = None
        self._cik_iids = None

    def __auto_check_cik__(self):
        if not self._checked and (self._cik.dts is None or self._cik.iid is None):
            raise NotImplementedError('cik(dts or iid) is not setup!')
        else:
            self._checked = True

    def _setup_cik(self, cik_dt_col: str, cik_iid_col: str):
        """
        设置 cik 列名
        :param cik_dt_col:
        :param cik_iid_col:
        :return:
        """

        self._cik = CIK(cik_dt_col, cik_iid_col)

    # def getDB(self, db):
    #     self.db = db

    def add_factor(self, db_table: str, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
                   cik_dt_format='datetime',

                   as_alias: (list, tuple, str) = None):
        conds = '1'  # not allow to set conds
        cik_dt, cik_iid = self.check_cik_dt(cik_dt=cik_dt, default_cik_dt=self._cik.dts), self.check_cik_iid(
            cik_iid=cik_iid, default_cik_iid=self._cik.iid)

        res = self._factors.add_factor(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
                                       cik_dt_format=cik_dt_format,
                                       conds=conds, as_alias=as_alias)
        if isinstance(res, tuple):
            self._factors.append(res)
        elif isinstance(res, list):
            self._factors.extend(res)
        else:
            raise ValueError('res is not list or tuple')

    def show_factors(self, reduced=False, to_df=True):
        return self._factors.show_factors(reduced=reduced, to_df=to_df)

    def __iter__(self):
        return self._factors.fetch_iter(self._node, self.cik_dt, self.cik_iid, reduced=True,
                                        add_limit=False)

    def head(self, reduced=True, ):
        """
        quick look top data
        :param reduced:
        :return:
        """

        return self.fetch(reduced=reduced, add_limit=True)

    def fetch(self, reduced=True, add_limit=False):
        if self._strict_cik:
            if self._cik_dts is None:
                raise KeyError('cik_dts is not setup!')
            if self._cik_iids is None:
                raise KeyError('cik_iids is not setup!')

        fetched = self._factors.fetch_iter(self._node, self.cik_dt, self.cik_iid, reduced=reduced,
                                           add_limit=add_limit)

        result = pd.concat(fetched, axis=1)
        columns = result.columns.tolist()

        return result

    @property
    def cik_dt(self):
        dt_format = "%Y%m%d"
        if self._cik_dts is None:
            return "  1 "
        else:
            cik_dts_str = "','".join(map(lambda x: x.strftime(dt_format), pd.to_datetime(self._cik_dts)))
            return f" toYYYYMMDD(cik_dt) in ('{cik_dts_str}') "

    def set_cik_dt(self, cik_dt: list):
        self._cik_dts = cik_dt

    @property
    def cik_iid(self):
        if self._cik_iids is None:
            return "  1 "
        else:
            cik_iid_str = "','".join(map(lambda x: x, self._cik_iids))
            return f" cik_iid in ('{cik_iid_str}') "

    def set_cik_iid(self, cik_iid: list):
        self._cik_iids = cik_iid
        pass


if __name__ == '__main__':
    pass
