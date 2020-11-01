# coding=utf-8

from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode


class DatabaseNotExists(Exception): pass


class Dumper(ClickHouseTableNode):
    def __init__(self, *args, **kwargs):
        super(Dumper, self).__init__(*args, **kwargs)

    def _dump_db(self, *dbs):
        for db in dbs:
            if db in self.databases:
                pass
            else:
                raise DatabaseNotExists(f'{db} is not exists')
        create_db_sql_list = [(f"CREATE DATABASE IF NOT EXISTS {db};\n\n", db) for db in dbs]
        return create_db_sql_list

    def _dump_table(self, db):
        h = []
        tables = self.query(f"show tables from {db}")
        if tables.empty:
            h.append('\n\n')
        else:
            tables = tables.values.ravel().tolist()
            for table in tables:
                sql = f"show create table {db}.{table}"
                create_sql = self.query(sql)['statement'].values.ravel().tolist()[0] + ';\n\n'
                h.append(create_sql)
        return h

    def _dump_structure(self, *dbs, outfile='create_structure.sql'):
        with open(outfile, 'w+') as f:
            for create_db_sql, db in self._dump_db(*dbs):
                sql_list = [create_db_sql] + self._dump_table(db)
                for sql in sql_list:
                    f.write(sql)

    def dump(self, *dbs, outfile='create_structure.sql', structure_only=True):
        if structure_only:
            pass
        else:
            raise ValueError('structure_only only accept True, False is not available yet!')
        self._dump_structure(*dbs, outfile=outfile)


if __name__ == '__main__':
    pass
