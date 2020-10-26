# coding=utf-8
import unittest

from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode
from ClickSQL.clickhouse.ClickHouse import DatabaseTypeError
from ClickSQL.conf.parse_rfc_1738_args import ArgumentError


class MyTestConnectionCase(unittest.TestCase):
    # def test_connection_success(self):
    #     node = ClickHouseTableNode(conn)
    #     df1 = node.tables
    #
    #     self.assertGreater(len(df1), 5)

    def test_connection_settings_failure_empty(self):
        conn = ""
        with self.assertRaises(ArgumentError):
            node = ClickHouseTableNode(conn)

    def test_connection_settings_failure_wrong_port(self):
        conn = "clickhouse://default:121231@10.3.10.1/syste"
        with self.assertRaises(TypeError):
            node = ClickHouseTableNode(conn)

        conn = "clickhouse://default:121231@10.3.10.1:sd/syste"
        with self.assertRaises(ValueError):
            node = ClickHouseTableNode(conn)

        conn = "clickhouse://default:121231@10.3.10.1:/syste"
        with self.assertRaises(ValueError):
            node = ClickHouseTableNode(conn)

    def test_connection_settings_failure_wrong_host(self):
        conn = "clickhouse://default:121231@10.2.3.1:10/syste"
        with self.assertRaises(TypeError):
            node = ClickHouseTableNode(conn)

    def test_connection_settings_failure_wrong_symbol(self):
        conn = "mysql://default:121231@99.99.9.9:8123/system"

        with self.assertRaises(DatabaseTypeError):
            node = ClickHouseTableNode(conn)

    def test_connection_failure(self):
        conn = "clickhouse://default:121231@99.99.9.9:8123/system"
        node = ClickHouseTableNode(conn)
        df1 = node.tables

        self.assertGreater(len(df1), 5)


if __name__ == '__main__':
    unittest.main()
