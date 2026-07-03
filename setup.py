# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:32:50 2026

@author: noahm
"""

from setuptools import setup, find_packages

# Setup function so srimpro can be installed as a library with pip
setup(
    name="srimpro",
    version="1.2",
    author="Noah Miggiani",
    author_email="noah.miggiani.srimpro@gmail.com",
    description="Automated calculating, plotting, and exporting of SRIM data for streamlined analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Noah-Miggs/srimpro",
    license="MIT",
    keywords='srim automation plotting material irradiation',
    packages=find_packages(),
    install_requires=["numpy>=2.4.0", "pandas>=3.0.3", "pysrim>=0.5.10"],
    python_requires=">=3.14.4",
)
