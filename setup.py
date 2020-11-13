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


# 获取依赖
def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(
    name="ClickSQL",
    version=__version__,
    python_requires='>=3.7.0',  # python环境
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
    setup_requires=read_requirements('requirements.txt'),  # 指定需要安装的依赖,

)
