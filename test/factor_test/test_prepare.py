# coding=utf-8
import unittest
import pandas as pd
import numpy as np
from ClickSQL.factor_test.factor_tools import detect_series_na


def generate_random_data(size=(100, 2), add_na=0):
    s = np.random.random(size=size)

    df = pd.DataFrame(s, columns=[f't{i}' for i in range(size[1])])
    na = pd.DataFrame([[None] * size[1]] * add_na, columns=[f't{i}' for i in range(size[1])])

    return df.append(na)


class MyTestCase_prepare(unittest.TestCase):
    def test_detect_series_na_nona(self):
        df = generate_random_data(size=(100, 1), add_na=0)
        na_value = detect_series_na(df['t0'])
        self.assertEqual(na_value, None)

    def test_detect_series_na_1na_noraise(self):
        df = generate_random_data(size=(100, 1), add_na=1)
        na_value = detect_series_na(df['t0'], raise_error=False)
        self.assertEqual(na_value, 1)

    def test_detect_series_na_2na_raise(self):
        df = generate_random_data(size=(100, 1), add_na=2)
        with self.assertRaises(ValueError):
            na_value = detect_series_na(df['t0'], raise_error=True)
            # self.assertEqual(na_value, 2)


if __name__ == '__main__':
    unittest.main()
