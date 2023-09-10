# Commonly used built-in imports
import datetime as dt
from functools import reduce
from glob import glob
import itertools as it
import os
from pathlib import Path

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
from mobilkit.spark import Spark
from mobilkit.geo import CRS_DEG, CRS_M
from mobilkit.gps import UID, LON, LAT, TS, ERR

# Display settings
U.config_display(disp_method=True)
plt.rcParams.update(U.MPL_RCPARAMS)
mpl.rcParams.update(U.MPL_RCPARAMS)

# Important paths
# common root directory of TNET-1, UMNI-2 & UMNI-5
UMNI = Path('/home/umni2/a/umnilab')
# base folder containing cleaned Quadrant ping data
QUADRANT = UMNI / 'data/quadrant'
# base folder containing SafeGraph's POI data
SAFEGRAPH = UMNI / 'data/safegraph'
# root directory for all projects
MK = UMNI / 'users/verma99/mk'
# Python interpreter for UMNI projects
MK_PYTHON = UMNI / 'users/verma99/anaconda3/envs/mk3.9/bin/python'

# Pyspark session handler
# configuration depends on the host server of the script
SERVER = os.uname().nodename.split('.')[0]
# set the handler
os.environ['PYSPARK_PYTHON'] = str(MK_PYTHON)
# Pyspark session handler with resources allocated 
# according to the host server
SP = Spark({k: v.get(SERVER, None) for k, v in {
    'executor.memory': dict(tnet1='200g', umni1='36g', umni2='36g', umni5='160g'),
    'driver.memory': dict(tnet1='200g', umni1='36g', umni2='36g', umni5='160g'),
    'default.parallelism': dict(tnet1=16, umni1=20, umni2=20, umni5=32)
}.items()} | {'local.dir': f'{MK}/.tmp'}, start=False)
# set the executor for this environment
# SP.context.pythonExec = str(MK_PYTHON)
