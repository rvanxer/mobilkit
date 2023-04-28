# Commonly used built-in imports
import datetime as dt
from functools import reduce
from glob import glob
import itertools as it
import os
from pathlib import Path
import warnings

# Commonly used external imports
import geopandas as gpd
from geopandas import GeoDataFrame as Gdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as Arr
import pandas as pd
from pandas import DataFrame as Pdf
from pandas import Series as Seq
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as Sdf
import seaborn as sns
from tqdm.notebook import tqdm

# Imports from mobilkit
import mobilkit as mk
from mobilkit import utils as U
from mobilkit.spark import Types as T
from mobilkit.geo import CRS_DEG, CRS_M
from mobilkit.gps import (UID, LON, LAT, TS, ERR)

# Display settings
mk.utils.config_display(disp_method=True)
plt.rcParams.update(mk.utils.MPL_RCPARAMS)
mpl.rcParams.update(mk.utils.MPL_RCPARAMS)

# Pyspark session handler
# configuration depends on the host server of the script
SERVER = os.uname().nodename.split('.')[0]
# Pyspark session handler with resources allocated 
# according to the host server
SP = mk.spark.Spark({k: v.get(SERVER, None) for k, v in {
    'executor.memory': dict(tnet1='200g', umni1='36g', umni2='36g', umni5='160g'),
    'driver.memory': dict(tnet1='200g', umni1='36g', umni2='36g', umni5='160g'),
    'default.parallelism': dict(tnet1=16, umni1=20, umni2=20, umni5=32)
}.items()}, start=False)

# Important paths
# common root directory of TNET-1, UMNI-2 & UMNI-5
UMNI = Path('/home/umni2/a/umnilab')
# base folder containing cleaned Quadrant ping data
QUADRANT = UMNI / 'data/Quadrant'
# base folder containing SafeGraph's POI data
SAFEGRAPH = UMNI / 'data/SafeGraph'
# root directory for all projects
MK = UMNI / 'users/verma99/mk'

# Project setup
class Project:
    def __init__(self, root):
        # convert the supplied project root path into absolute path
        self.root = root = Path(root).resolve()
        # resolve the name of this project
        self.name = root.stem.replace('_', ' ').title()
        # directory containing all the relevant project-specific data
        self.data = U.mkdir(root / 'data')
        # directory where the output figures are stored
        self.fig = root / 'fig'
        # create a parameters file
        self.params = U.Params(self.root / 'params')

    def __repr__(self):
        return f'Project("{self.name}")'

    def imsave(self, *args, **kwargs):
        # customize the `mobilkit.utils.plot()` function with
        # this project's image output directory path
        root = Path(kwargs.pop('root', self.fig))
        U.imsave(*args, root=root, **kwargs)
