from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = "Rajat's mobilkit implementation in pyspark"
LONG_DESCRIPTION = 'A package containing helper tools for processing smartphone GPS data, US census data and geospatial data.'

# Setting up
setup(
    name="mk",
    version=VERSION,
    author="Rajat Verma",
    author_email="rajatverma1995@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pyspark', 'numpy', 'pandas', 'geopandas', 'osmnx'],
    keywords=['python', 'gps', 'gis', 'mobility', 'spark'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
