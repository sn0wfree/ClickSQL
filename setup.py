# coding=utf-8
from setuptools import setup, find_packages

from ClickSQL import __version__, __author__

setup(
    name="ClickSQL",
    version=__version__,
    keywords=("ClickSQL", "Databases"),
    description="SQL programming",
    long_description="clickhouse databases",
    license="MIT Licence",

    url="http://www.github.com/sn0wfree",
    author=__author__,
    author_email="snowfreedom0815@gmail.com",

    packages=find_packages(),
    include_package_data=True


)
