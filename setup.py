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
            if not line.startswith('#') and line.strip()]


setup(
    name="ClickSQL",
    version=__version__,
    python_requires='>=3.8',
    keywords=("ClickHouse", "Databases", "SQL", 'Python', 'Client', 'MCP', 'Async'),
    description="A Python client for ClickHouse with async and MCP support",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    license="MIT Licence",

    url="http://www.github.com/sn0wfree/ClickSQL",
    author=__author__,
    author_email="snowfreedom0815@gmail.com",

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Database',
    ],

    packages=find_packages(exclude=['dist', 'docs', 'ClickSQL.egg-info', 'test', 'examples']),
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),

    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.0.0',
        ],
    },

    entry_points={
        'console_scripts': [
            'clicksql-mcp=ClickSQL.mcp.server:main',
        ],
    },
)
