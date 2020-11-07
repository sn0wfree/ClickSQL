# coding=utf-8

from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode

from ClickSQL.errors import DatabaseNotExists


class Dumper(ClickHouseTableNode):
    def __init__(self, *args, **kwargs):
        super(Dumper, self).__init__(*args, **kwargs)

    def _dump_db_structure(self, *dbs: str):
        """

        :param dbs:
        :return:
        """
        res = filter(lambda x: x not in self.databases, dbs)
        for db in res:
            raise DatabaseNotExists(f'{db} is not exists')
        create_db_sql_list = [(f"CREATE DATABASE IF NOT EXISTS {db};\n\n", db) for db in dbs]
        return create_db_sql_list

    def _dump_table_structure(self, db: str, table=None):
        """

        :param db:
        :param table:
        :return:
        """
        h = []
        if table is None:
            tables = self.query(f"show tables from {db}")
            if tables.empty:
                h.append('\n\n')
            else:
                tables = tables.values.ravel().tolist()
                for table in tables:
                    sql = f"show create table {db}.{table}"
                    create_sql = self.query(sql)['statement'].values.ravel().tolist()[0] + ';\n\n'
                    h.append(create_sql)
        else:
            if isinstance(table, str):
                sql = f"show create table {db}.{table}"
                create_sql = self.query(sql)['statement'].values.ravel().tolist()[0] + ';\n\n'
                h.append(create_sql)
            else:
                raise ValueError('table parameter get wrong type!')
        return h

    def _dump_structure(self, *dbs, outfile='create_structure.sql'):
        """

        :param dbs:
        :param outfile:
        :return:
        """
        with open(outfile, 'w+') as f:
            for create_db_sql, db in self._dump_db_structure(*dbs):
                sql_list = [create_db_sql] + self._dump_table_structure(db)
                for sql in sql_list:
                    f.write(sql)

    def dump(self, *dbs, outfile='create_structure.sql', structure_only=True):
        """

        :param dbs:
        :param outfile:
        :param structure_only:
        :return:
        """
        if structure_only:
            self._dump_structure(*dbs, outfile=outfile)
        else:
            raise ValueError('structure_only only accept True, False is not available yet!')


if __name__ == '__main__':
    conn = "clickhouse://default:Imsn0wfree@47.104.186.157:8123/system"
    c = Dumper(conn)._dump_db_structure('system', 'raw')
    print(c)
    pass
