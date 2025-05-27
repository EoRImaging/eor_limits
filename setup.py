# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""eor_limits setup."""

import io
from setuptools import setup

version = 0.1

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "eor_limits",
    "author": "Nichole Barry, Bryna Hazelton",
    "url": "https://github.com/EoRImaging/eor_limits",
    "license": "BSD",
    "description": "a small utility for plotting EoR Limits",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"eor_limits": "eor_limits"},
    "packages": ["eor_limits", "eor_limits.data"],
    "scripts": ["eor_limits/plot_eor_limits.py"],
    "version": version,
    "include_package_data": True,
    "install_requires": ["numpy", "matplotlib", "pyyaml", "h5py"],
    "extras_require": {"dev": ["pytest", "pytest-cov", "coverage", "pre-commit"]},
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "cosmology",
}

if __name__ == "__main__":
    setup(**setup_args)
