# coding: utf-8

"""
@File   : setup.py.py
@Author : garnet
@Time   : 2020/8/11 17:57
"""

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = [package.strip() for package in f.readlines()]

setup(
    name='garnet',
    version='0.0.1',
    description='Toolkit for deep learning especially NLP tasks on keras platform',
    keywords=['garnet', 'NLP', 'machine learning', 'deep learning', 'keras', 'preprocess'],
    author='garnetow',
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
