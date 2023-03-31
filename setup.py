# coding: utf8
"""
Setup script for bancs
============================
This script allows to install BANCS within the Python environment.
Usage
-----
::
    pip install -e .
"""

import re
import os
from setuptools import setup, find_packages

# Get the version from __init__.py
path = os.path.join(os.path.dirname(__file__), 'bancs', '__init__.py')
with open(path) as f:
    version_file = f.read()

version = re.search(r"^\s*__version__\s*=\s*['\"]([^'\"]+)['\"]",
                    version_file, re.M)
if version:
    version = version.group(1)
else:
    raise RuntimeError("Unable to find version string.")

# Long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bancs',
    version=version,
    license='LGPLv3+',
    author="Elias Fekhari",
    author_email='elias.fekhari@edf.fr',
    packages=['bancs'],
    url='https://github.com/efekhari27/bancs',
    keywords=['OpenTURNS', 'Reliability'],
    description="Bernstein Adaptive Nonparametric Conditional Sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
          "pandas",
          "matplotlib",
          "openturns>=1.17"
      ],
    include_package_data=True,
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],

)