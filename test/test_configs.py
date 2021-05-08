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
        self.assertIsInstance(Config.get('test'), (list,))

    def test_Configs_list(self):
        Config.set('test', ['1', '2'])
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), (list,))

    def test_Configs_dict(self):
        Config.set('test', {'1': 1, '2': 2})
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), (dict,))

    # def test_Configs_set(self):
    #     Config.set('test', {'1', '2'})
    #     print(Config.get('test'))
    #     self.assertIsInstance(Config.get('test'), (set,))

    def test_Configs_bool(self):
        Config.set('test', True)
        print(Config.get('test'))
        self.assertIsInstance(Config.get('test'), bool)
        Config.set('test2', False)
        print(Config.get('test2'))
        self.assertIsInstance(Config.get('test2'), bool)


if __name__ == '__main__':
    unittest.main()
