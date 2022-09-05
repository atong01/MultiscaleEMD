from setuptools import find_packages
from setuptools import setup

import os

install_requires = [
    "numpy>=1.16.0",
    "scipy",
    "matplotlib>=3.0",
    "pot",
    "pygsp",
    "graphtools",
    "scikit-learn>=1.0.0",
]

doc_requires = [
    "sphinx",
    "sphinxcontrib-napoleon",
    "ipykernel",
    "nbsphinx",
    "autodocsumm",
]

test_requires = [
    "pandas",
    "black",
    "flake8",
    "pytest",
]

version_py = os.path.join(os.path.dirname(__file__), "MultiscaleEMD", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.rst").read()

setup(
    name="MultiscaleEMD",
    packages=find_packages(),
    version=version,
    description="Multiscale approximations to the earth mover's distance.",
    author="Alexander Tong",
    author_email="alexandertongdev@gmail.com",
    license="MIT",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "doc": doc_requires,
    },
    long_description=readme,
    url="https://github.com/atong01/MultiscaleEMD",
)
