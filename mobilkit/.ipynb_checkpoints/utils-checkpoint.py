"""
Miscellaneous utility functions for file handling, plotting, display, etc.

@created May 9, 2022
"""
from datetime import datetime, date
import json
import os
from pathlib import Path
import re
import warnings

import geopandas as gpd
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
import pyspark
import seaborn
import yaml

# Unit conversion factors
MI2M = 1609.34  # mile to meter
M2MI = 1 / 1609.34  # meter to mile
MI2KM = 1.60934  # mile to kilometer
KM2MI = 1 / 1.60934  # kilometer to mile
SQMI2SQM = 2.58998811e6  # sq. mile to sq. meter
SQM2SQMI = 1 / 2.58998811e6  # sq. meter to sq. mile

# matplotlib custom plotting parameters
MPL_RCPARAMS = {
    'figure.titlesize': 15,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'legend.title_fontsize': 13,
    'figure.dpi': 100
}


class Params:
    """
    A class for defining and updating configuration settings
    and parameters in a YAML file.
    """

    def __init__(self, path):
        self.path = Path(str(path) + '.yaml')
        self._data = {}
        self.read()

    def __repr__(self):
        return f'Params({self.path})'

    def get(self, key_str, sep='.'):
        keys = key_str.split(sep)
        node = self._data
        for key in keys:
            if key in node:
                node = node[key]
            else:
                raise KeyError(f'"{key}" of "{key_str}" not found in params.')
        return node

    def set(self, values, write=True):
        self._data = update_nested_dict(self._data, values)
        if write:
            self.write()
            
    def read(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self._data = yaml.safe_load(f)
    
    def write(self, indent=4, sort=False):
        with open(self.path, 'w') as f:
            yaml.dump(self._data, f, indent=indent,
                      allow_unicode=True, sort_keys=sort)


def mkdir(path):
    """
    Shorthand for making a folder if it does not exist.

    Parameters
    ----------
    path : str | Path
        Folder to be created (if it does not exist).
    
    Returns
    -------
    Path
        Same path as input but converted to a PosixPath.
    """
    assert isinstance(path, str) or isinstance(path, Path)
    Path(path).mkdir(exist_ok=True, parents=True)
    return Path(path)


def update_nested_dict(d, u):
    """
    Update a given base nested dictionary with another (possibly nested) 
    dictionary without modifying the other levels of the base dictionary.
    Solution taken from https://stackoverflow.com/a/3233356/5711244
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def ignore_warnings(func):
    """
    A decorator to remove warnings in functions which produce safe warnings
    (e.g., in some geopandas functions). Only to be used for aesthetic reasons.
    """
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = func(*args, **kwargs)
        return result
    return wrapper


def log(msg):
    """
    Print a message with timestamp.
    """
    print(f'[{datetime.now()}] {msg}')
    

def str2dt(date_str):
    """
    Convert a date string in the 'YYYY-MM-DD' format to a datetime time object.
    """
    return date(*[int(x) for x in date_str.split('-')])


def dates(start: str, end: str):
    return pd.date_range(start, end).date


def normalize(x):
    """
    Normalize an array of values to fit in the range [0, 1].
    """
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standardize(x, err=1e-10):
    """
    Standardize an array of values (i.e., get the z-scores).
    """
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    return (x - x.mean()) / (x.std() + err)


def write_json(obj, path, suffix='.params'):
    """
    Write a dictionary as a JSON file, usually to show the parameters
    associated with the preparation of a dataset.

    Parameters
    ----------
    obj : dict
        Object to be written.
    path : str
        Path of the output file without the extension.
    suffix : str
        If needed, specify kind of JSON. For example, a ".params.json" file
        contains parameters of a dataset operation.

    Returns
    -------
    None
    """
    # create parent folder if it does not exist
    outdir = Path(path).parent.mkdir(exist_ok=True, parents=True)
    # add the suffix to the JSON file name
    outfile = str(Path(outdir, path.stem + suffix + '.json'))
    # write the object
    with open(outfile, 'w') as f:
        f.write(json.dumps(obj))


def disp(x, top=1, mem=True, vert=False):
    """
    Custom display for pandas and geopandas dataframe and series objects in
    Jupyter. This is a combination of methods like `head`, `dtypes`, and
    `memory_usage`.

    Parameters
    ----------
    x : pd.DataFrame | pd.Series | gpd.GeoDataFrame | gpd.GeoSeries |
        pyspark.sql.DataFrame
        Data object to be pretty displayed.
    top : int
        No. of first rows to be displayed.
    mem : bool
        Whether show size of memory usage of the object (in MiB). May be turned
        off for heavy objects whose memory consumption computation may take time.
    vert : bool-like
        Whether use vertical layout for pyspark.sql.DataFrame display.
    Returns
    -------
    The input object as it is.
    """
    def f(tabular: bool, crs: bool, mem=mem):
        """
        tabular : Is the object `x` a 2D (matrix or table-like) structure?
        crs : Does the object `x` have a CRS, i.e., is it geographic/geometric?
        """
        shape = ('{:,} rows x {:,} cols'.format(*x.shape) if tabular
                 else f'{x.size:,} rows')
        mem = x.memory_usage(deep=True) / (1024 ** 2) if mem else None
        memory = f'Memory: {(mem.sum() if tabular else mem):.1f} MiB'
        crs = f'CRS: {x.crs.srs}' if crs else ''
        print(shape + '; ' + memory + ('; ' + crs if crs else ''))
        if tabular:
            types = {x.index.name or '': '<' + x.dtypes.astype(str) + '>'}
            types = pd.DataFrame(types).T
            display(pd.concat([types, x.head(top).astype({'geometry': str}) if
                    crs else x.head(top)]))
        else:
            print(x.head(top))

    if isinstance(x, gpd.GeoDataFrame):
        f(True, True)
    elif isinstance(x, pd.DataFrame):
        f(True, False)
    elif isinstance(x, gpd.GeoSeries):
        f(False, True)
    elif isinstance(x, pd.Series):
        f(False, False)
    elif isinstance(x, pyspark.sql.DataFrame):
        if top == 0:
            x.printSchema()
        else:
            x.show(top, vertical=bool(vert))
    return x


# noinspection PyDefaultArgument
def config_display(
        disp_method=False,
        sns_style='whitegrid',
        pd_options=(('display.max_columns', 100),),
        mpl_options=MPL_RCPARAMS):
    """
    Set default display and plotting settings for Jupyter/IPython environment.

    Parameters
    ----------
    disp_method : bool
        Add `util.disp()` as `pandas` and `geopandas` methods for easy
        visualization during exploration (not recommended for use in serious
        projects, only for private use).
    pd_options : tuple[tuple[str, Any]]
        Pandas dataframe display options in Jupyter.
    sns_style : str
        Default theme style of seaborn plots.
    mpl_options : tuple[tuple[str, Any]]
        Default matplotlib plotting parameters.
    Returns
    -------
    None
    """
    seaborn.set_style(sns_style)
    for opt in pd_options:
        pd.set_option(*opt)
    mpl.rcParams.update(mpl_options)
    if disp_method:
        pd.DataFrame.disp = disp
        gpd.GeoDataFrame.disp = disp
        gpd.GeoSeries.disp = disp
        pd.Series.disp = disp
        pyspark.sql.DataFrame.disp = disp


def imsave(title=None, fig=None, ax=None, dpi=200, root='./fig', ext='png', opaque=True):
    """
    Save the current matplotlib figure to disk
    Parameters
    ----------
    title : str
        Title of the filename (may contain special characters that will be
        removed).
    root : Path | str
        Folder path where the image is to be saved.
    fig : plt.Figure
        The figure which needs to be saved. If None, use the current figure.
    ax : plt.Axes
        Axes object of interest.
    dpi : int
        Dots per inch (quality) of the output image.
    ext : str
        File extension: One of supported types like 'png' and 'jpg'.
    opaque : bool
        Whether the output is to be opaque (if extension supports transparency).

    Returns
    -------
    None
    """
    fig = fig or plt.gcf()
    ax = ax or fig.axes[0]
    title = title or fig._suptitle or ax.get_title() or 'Untitled {}'.format(
        datetime.now().strftime('%Y-%m-%d_%H-%m-%S'))
    title = re.sub(r'[^A-Za-z\s\d,.-]', '_', title)
    fig.savefig(f'{mkdir(root)}/{title}.{ext}', dpi=dpi, bbox_inches='tight',
                transparent=not opaque, facecolor='white' if opaque else 'auto')


def plot(ax=None, fig=None, size=None, dpi=100, title=None, xlab=None,
         ylab=None, xlim=None, ylim=None, titlesize=None, xlabsize=None,
         ylabsize=None, xeng=False, yeng=False, xticks=None, yticks=None,
         xticks_rotate=None, yticks_rotate=None, xlog=False, ylog=False,
         axoff=False, gridcolor=None, framebordercolor=None,
         save=False, path=None):
    """
    Custom handler for matplotlib plotting options.

    Parameters
    ----------
    ax : plt.Axes
    fig : plt.Figure
    size : tuple[float, float]
        Figure size (width x height) (in inches).
    dpi : float
        Figure resolution measured in dots per inch (DPI).
    title, xlab, ylab : str
        Axes title, x-axis label and y-axis label.
    titlesize, xlabsize, ylabsize : float
        Font size of the axes title, xlabel, and ylabel.
    xlim, ylim : tuple[float, float]
        X-axis and y-axis lower and upper limits.
    xeng, yeng : bool
        Whether x/y-axis ticks are to be displayed in engineering format.
    xtime : bool
        Whether x-axis is to be displayed as time series.
    xticks, yticks: list-like
        Tick markers.
    xticks_rotate, yticks_rotate : float
        Extent of rotation of xticks/yticks (in degrees).
    xlog, ylog : bool
        Whether x/y-axis is to be displayed on log_10 scale.
    axoff : bool
        Whether turn off the axis boundary.
    gridcolor : str
        Color of the gridlines if a grid is shown.
    framebordercolor : str
        Color of the plotting frame's border.
    save : bool
        Whether the plotted image is to be saved to disk.
    path : str
        Path where the image is to be saved.

    Returns
    -------
    ax : plt.Axes
    """
    if isinstance(size, tuple) and fig is None:
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
    ax = ax or plt.gca()
    ax.set_title(title, fontsize=titlesize or mpl.rcParams['axes.titlesize'])
    ax.set_xlabel(xlab, fontsize=xlabsize or mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(ylab, fontsize=ylabsize or mpl.rcParams['axes.labelsize'])
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if xeng: ax.xaxis.set_major_formatter(EngFormatter())
    if yeng: ax.yaxis.set_major_formatter(EngFormatter())
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    if xticks: ax.set_xticks(xticks)
    if yticks: ax.set_yticks(yticks)
    if xticks_rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticks_rotate)
    if yticks_rotate:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=yticks_rotate)
    if axoff: ax.axis('off')
    if gridcolor: ax.grid(color=gridcolor)
    if framebordercolor:
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_color(framebordercolor)
    fig = fig or plt.gcf()
    auto_title = 'Untitled-' + datetime.now().isoformat().replace(':', '-')
    if save: imsave(title or auto_title, root=mkdir(path), fig=fig)
    return ax
