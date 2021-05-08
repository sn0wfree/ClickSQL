import unittest
from ClickSQL.conf import Config


class MyTestCaseConfigs(unittest.TestCase):
    def test_Configs_int(self):
        Config.set('test', 1)
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), int)

    def test_Configs_str(self):
        Config.set('test', '1')
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), str)

    # def test_Configs_byte(self):
    #     Config.set('test', b'1')
    #     print(Config.get('test'))
    #     self.assertIsInstance(Config.get('test'), str)

    def test_Configs_tuple(self):
        Config.set('test', ('1', '2'))
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), (tuple, list))

    def test_Configs_list(self):
        Config.set('test', ['1', '2'])
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), (tuple, list))


if __name__ == '__main__':
    unittest.main()
