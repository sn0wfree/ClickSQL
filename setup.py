# coding=utf-8
import os

from setuptools import setup, find_packages

from ClickSQL import __version__, __author__

# 读取文件内容
this_directory = os.path.abspath(os.path.dirname(__file__))


def read_file(filename):
    with open(os.path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


setup(
    name="ClickSQL",
    version=__version__,
    keywords=("ClickSQL", "Databases"),
    description="SQL programming",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",  # 新参数
    license="MIT Licence",

    url="http://www.github.com/sn0wfree/ClickSQL",
    author=__author__,
    author_email="snowfreedom0815@gmail.com",

    packages=find_packages(exclude=['dist', 'doc', 'ClickSQL.egg-info']),
    include_package_data=True,
    install_requires=['pandas>=1.0.5',
                      'numpy>=1.18.5',
                      'requests>=2.20.0',
                      'aiohttp>=3.6.2',
                      'nest-asyncio>=1.4.1'
                      ],

)
