from glob import glob
import itertools as it
import json
import operator as op
import os
from pathlib import Path
import shutil
import urllib
import warnings

import geopandas as gpd
from geopandas import GeoDataFrame as Gdf
import haversine as hs
import numpy as np
import pandas as pd
import requests
import scipy
import sklearn

# Global coordinate reference systems (CRS) for uniformity (chosen arbitrarily).
# spatial CRS with coordintates in meters
CRS_M = 'EPSG:3857'
# geographical CRS with coordintatges in degrees
CRS_DEG = 'EPSG:4326'

# Schema of SafeGraph places
# Description on https://docs.safegraph.com/docs/places#places-schema
SG_POIS_SCHEMA = {
    'placekey': 'str',
    'parent_placekey': 'str',
    'location_name': 'str',
    'safegraph_brand_ids': 'str',
    'brands': 'str',
    'top_category': 'str',
    'sub_category': 'str',
    'naics_code': 'int32',
    'latitude': 'float',
    'longitude': 'float',
    'street_address': 'str',
    'city': 'str',
    'region': 'str',
    'postal_code': 'int32',
    'iso_country_code': 'str',
    'phone_number': 'str',
    'open_hours': 'str',
    'category_tags': 'str',
    'opened_on': 'str',
    'closed_on': 'str',
    'tracking_closed_since': 'str',
    'geometry_type': 'str'
}

# Some important industry categories and their NAICS codes (defined arbitrarily)
IMP_NAICS = [
    # Code   Category               Official name in the NAICS table
    # -----  --------               --------------------------------
    (445110, 'Supermarkets',        'Supermarkets and Other Grocery (except Convenience) Stores'),
    (447110, 'Gas stations',        'Gasoline Stations with Convenience Stores'),
    (531120, 'Malls',               'Lessors of Nonresidential Buildings (except Miniwarehouses)'),
    (611110, 'Schools',             'Elementary and Secondary Schools'),
    (622110, 'Hospitals',           'General Medical and Surgical Hospitals'),
    (624410, 'Daycare centers',     'Child Day Care Services'),
    (712190, 'Nature parks',        'Nature Parks and Other Similar Institutions'),
    (713940, 'Fitness centers',     'Fitness and Recreational Sports Centers'),
    (721110, 'Hotels/Motels',       'Hotels (except Casino Hotels) and Motels'),
    (722410, 'Bars/Pubs',           'Drinking Places (Alcoholic Beverages)'),
    (722511, 'Full Restaurants',    'Full-Service Restaurants'),
    (722513, 'Fast food/Takeout',   'Limited-Service Restaurants'),
    (722515, 'Coffee/Snack places', 'Snack and Nonalcoholic Beverage Bars'),
]

# Custom categorization of OSM POI classes into broad categories
OSM_CLASSES = {
    'education':  ['kindergarten', 'library', 'school'],
    'food_drink': ['beverages', 'fast_food', 'restaurant'],
    'hotel':      ['hotel', 'motel'],
    'other':      ['arts_centre', 'artwork', 'fountain', 'tower', 'viewpoint',
                   'water_tower'],
    'medical':    ['dentist'],
    'recreation': ['dog_park', 'ice_rink', 'park', 'picnic_site', 'pitch',
                   'playground', 'sports_centre', 'stadium', 'swimming_pool',
                   'track'],
    'shopping':   ['bookshop', 'convenience', 'mall', 'supermarket'],
    'utilities':  ['atm', 'bank', 'community_centre', 'fire_station',
                   'hairdresser', 'shelter', 'toilet', 'town_hall',
                   'water_works'],
}

# FIPS codes of all the 50 US states
# obtained from https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt
US_STATES_FIPS = {
    'ALABAMA': 1,
    'ALASKA': 2,
    'ARIZONA': 4,
    'ARKANSAS': 5,
    'CALIFORNIA': 6,
    'COLORADO': 8,
    'CONNECTICUT': 9,
    'DELAWARE': 10,
    'DISTRICT OF COLUMBIA': 11,
    'FLORIDA': 12,
    'GEORGIA': 13,
    'HAWAII': 15,
    'IDAHO': 16,
    'ILLINOIS': 17,
    'INDIANA': 18,
    'IOWA': 19,
    'KANSAS': 20,
    'KENTUCKY': 21,
    'LOUISIANA': 22,
    'MAINE': 23,
    'MARYLAND': 24,
    'MASSACHUSETTS': 25,
    'MICHIGAN': 26,
    'MINNESOTA': 27,
    'MISSISSIPPI': 28,
    'MISSOURI': 29,
    'MONTANA': 30,
    'NEBRASKA': 31,
    'NEVADA': 32,
    'NEW HAMPSHIRE': 33,
    'NEW JERSEY': 34,
    'NEW MEXICO': 35,
    'NEW YORK': 36,
    'NORTH CAROLINA': 37,
    'NORTH DAKOTA': 38,
    'OHIO': 39,
    'OKLAHOMA': 40,
    'OREGON': 41,
    'PENNSYLVANIA': 42,
    'RHODE ISLAND': 44,
    'SOUTH CAROLINA': 45,
    'SOUTH DAKOTA': 46,
    'TENNESSEE': 47,
    'TEXAS': 48,
    'UTAH': 49,
    'VERMONT': 50,
    'VIRGINIA': 51,
    'WASHINGTON': 53,
    'WEST VIRGINIA': 54,
    'WISCONSIN': 55,
    'WYOMING': 56
}


class Bbox:
    """ A geographic rectangle that serves as a bounding box. """

    def __init__(self, left=None, bottom=None, right=None, top=None,
                 unit='deg', geom=None):
        """
        Parameters
        ----------
        left, bottom, right, top : float
            Coordinates (degrees) of the edges of the rectangle.
        geom : geopandas.GeoSeries | geopandas.GeoDataFrame
        """
        if isinstance(geom, gpd.GeoSeries):
            left, bottom, right, top = geom.unary_union.bounds
        if isinstance(geom, gpd.GeoDataFrame):
            left, bottom, right, top = geom.geometry.unary_union.bounds
        self.coords = (left, bottom, right, top)
        self.left, self.bottom, self.right, self.top = self.coords
        self.unit = unit
        self.width = right - left
        self.height = top - bottom
        self.cx = cx = (left + right) / 2
        self.cy = cy = (top + bottom) / 2
        self.center = (cx, cy)

    def __repr__(self):
        return ('Rect(W={left:.6f}, N={top:.6f}, E={right:.6f}, '
                'S={bottom:.6f}, unit={unit})').format(**vars(self))


def pdf2gdf(df, x='lon', y='lat', crs=None):
    """
    Convert a pandas DataFrame to a geopandas GeoDataFrame by creating point
    geometry from the dataframes x & y columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be converted.
    x, y : str
        Column names corresponding to X (longitude) & Y (latitude) coordinates.
    crs : str | int | None
        Coordinate reference system of the converted dataframe.
    Returns
    -------
    gpd.GeoDataFrame
        Converted GDF.
    """
    geom = gpd.points_from_xy(df[x], df[y], crs=crs)
    return gpd.GeoDataFrame(df, geometry=geom)


def gdf2pdf(df, x='lon', y='lat', crs=None):
    """
    Unpack the coordinates of a Point geometry sequence object to a pandas
    dataframe.

    Parameters
    ----------
    df : gdf.GeoDataFrame | gpd.GeoSeries
        Dataframe or series whose geometry column will be used to get coords.
        Must be all Point or MultiPoint features in the geometry column.
    x, y : str
        Column names corresponding to X (longitude) & Y (latitude) coordinates.
    crs : str | int | None
        Optionally convert :attr:`df` to this CRS before getting coordinates.
    Returns
    -------
    pd.DataFrame
        Converted dataframe with just 2 columns for X & Y.
    """
    if isinstance(crs, str) or isinstance(crs, int):
        df = df.to_crs(crs)
    geom = df if isinstance(df, gpd.GeoSeries) else df.geometry
    return pd.DataFrame(geom.apply(lambda g: g.coords[0]).tolist(),
                        columns=[x, y])


def dist_m2deg(dist, lat, a=6_371_001, b=6_356_752):
    """
    Convert given distance in meters to degrees at a point with given latitude
    by computing the meridional radius of curvature.
    Source: https://en.wikipedia.org/wiki/Earth_radius#Meridional

    Parameters
    ----------
    dist : float
        Input distance in meters.
    lat : float
        Latitude of the point at which conversion is to take place.
    a, b : float
        Radii of Earth at equator & poles respectively (meters).
    Returns
    -------
    float
        Distance in degrees.
    """
    lat = np.deg2rad(lat)
    curve_radius = (a * b) ** 2 / (
            (a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2) ** 1.5
    return np.rad2deg(dist / curve_radius)


def get_dist(x0, y0, x1, y1, df=None, haversine=True, unit='m'):
    """
    Compute the pairwise distances between two sets of points. If points are in
    degrees, Haversine distance is recommended, but if they are in a plane, then
    Euclidean distance is recommended.

    Parameters
    ----------
    x0, y0, x1, y1 : str | iterable[float]
        Either the vectors (list-like) containing the x & y coordinates of the
        two point sets or the names of the columns if `df` is given.
    df : pandas.DataFrame
        Optional dataframe containing the `x0, y0, x1, y1` columns.
    haversine : bool
        If true, compute Haversine distance, otherwise Euclidean distance.
    unit : str
        If `haversine==True`, report the result in this unit.

    Returns
    -------
    numpy.array
        A vector containing pairwise distances between the two point sets.
    """
    assert type(x0) == type(y0) == type(x1) == type(y1)
    if isinstance(x0, str):
        x0, y0, x1, y1 = df[x0], df[y0], df[x1], df[y1]
    if haversine:
        return hs.haversine_vector(list(zip(y0, x0)), list(zip(y1, x1)), unit=unit)
    else:
        p0 = np.vstack([np.array(x0), np.array(y0)]).T
        p1 = np.vstack([np.array(x1), np.array(y1)]).T
        return np.linalg.norm(p0 - p1, axis=1, ord=2)


def intersect_polygon_area_map(src, trg):
    """
    Compute a matrix of intersection areas (in sq. mi.) between two overlapping
    polygon geometry layers.

    Parameters
    ----------
    src, trg : geopandas.GeoDataFrame
        Input tables whose index contains the feature IDs.

    Returns
    -------
    pandas.DataFrame
        A matrix of shape (n1, n2) where n1 & n2 are the no. of rows of src &
        trg respectively.
    """
    src = src.to_crs(CRS_M)[['geometry']].rename_axis('src_fid').reset_index()
    trg = trg.to_crs(CRS_M)[['geometry']].rename_axis('trg_fid').reset_index()
    res = gpd.overlay(src, trg, how='intersection', keep_geom_type=False)
    res['area'] = res.pop('geometry').area / 2.59e6
    return res.pivot('src_fid', 'trg_fid', 'area').fillna(0)


def meanshift(data, bandwidth=None, bin_seeding=True, include_orphans=True):
    """
    Custom wrapper around Scikit Learn's MeanShift clustering implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.cluster
    .MeanShift.html).

    Parameters
    ----------
    data : numpy.array
        Input data matrix.
        Shape: (`num_samples`, `num_features`).
    bandwidth : float
        Bandwidth of RBF kernel to be used in MeanShift. If the lat and lon are
        in degrees, this value needs to be in degrees, not regular distance
        units. A reasonable approximation can be found using meter-to-degree
        conversion in `mobilkit.geo.dist_m2deg` for which the approximate mean
        latitude of the target place needs to be given.
        CAUTION: If this parameter is not given, the algorithm can take
        sufficiently more time as described in the documentation.
    bin_seeding : bool
        Whether sample from grids of size :attr:`bandwidth` to generate the
        seeds needed for the algorithm. If set to false, runtime may increase
        significantly.
    include_orphans : bool
        If true, include orphan points (those that did not get any cluster with
        the given bandwidth) in their closest clusters. If there are any major
        outliers, they can substantially influence the means of the clusters,
        so the decision for this flag is to be judicious. See the documentation
        for details.

    Returns
    -------
    np.array
        The matrix of all the cluster centers.
        Shape: (`num_clusters`, `num_features`).
    np.array
        The array of cluster label (ranging from 0 to `num_clusters`-1) for
        each sample. Shape: (`num_samples`).
    """
    try:
        model = sklearn.cluster.MeanShift(
            bandwidth=bandwidth, bin_seeding=bin_seeding, 
            cluster_all=include_orphans)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(data)
            return model.cluster_centers_, model.labels_
    except ValueError:
        return None, None


def meanshift_top(x, y, bw: float, kwargs: str):
    try:
        model = sklearn.cluster.MeanShift(bandwidth=bw, **json.loads(kwargs))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(np.vstack([x, y]).T)
        return model.cluster_centers_[0].tolist()
    except ValueError:
        return [np.nan, np.nan]


def get_tiger_shp(dataset, file, fips=0, year=2021, overwrite=False):
    """
    Download the TIGER/Line shapefile of US region(s) or load it from disk.

    Parameters
    ----------
    dataset : str
        The label of the dataset to be downloaded. The currently supported
        product names are:
            ---------------------------------------------------------------
            Layer           Description                  Available at scale
            ---------------------------------------------------------------
            bg              census block groups          state
            county          counties                     nation
            csa             combined statistical areas   nation
            edges           ???                          county
            place           places (???)                 county
            primary_roads   ???                          nation
            roads           all roads                    county
            rails           all rail lines (?)           nation
            state           states                       nation
            tract           census tracts                state
            zcta520         zip code tabulation areas    nation
    file : str
        Path of the output file. Should end in '.zip'.
    fips : int
        FIPS code of the region of interest.
        For states, it is at most a 2-digit number.
        For counties, it is at most a 5-digit number.
    year : int
        The year whose data are to be downloaded.
    overwrite : bool
        Whether overwrite a possibly existing output file.

    Returns
    -------
    df : geopandas.GeoDataFrame
        The response object of the URL's GET output.
    """
    file = Path(file)
    if file.exists() and not overwrite:
        return gpd.read_file(file)
    # aliases for spatial scales at which shapefiles exist
    us, state, cnty = 'us', f'{fips:02}', f'{fips:05}'
    # file identifier based on the product type
    geo_query = {'bg': state, 'county': us, 'csa': us, 'edges': cnty,
                 'place': cnty, 'primary_roads': us, 'roads': cnty, 'rails': us,
                 'state': us, 'tract': state, 'zcta520': us}[dataset]
    # build the URL
    url = (f'https://www2.census.gov/geo/tiger/TIGER{year}/'
           f'{dataset.upper()}/tl_{year}_{geo_query}_{dataset}.zip')
    # get the response of the request
    resp = requests.get(url, stream=True, allow_redirects=True)
    # if status is OK
    if resp.status_code == 200:
        # even with OK status, the file can still be empty (seems like some bug
        # in the TIGER/Line API), so skip it
        if resp.url == 'https://www.census.gov/404.html':
            return 'Error 404 not found: ' + resp.url
        # create the parent folder if it does not exist
        Path(file).parent.mkdir(exist_ok=True, parents=True)
        # write as compressed shapefile
        with open(file, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=512):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    else:
        raise ValueError(
            'File could not be downloaded using given request: ', resp)
    return gpd.read_file(file)


def nearest_point(pts, df, df_cols=[]):
    """
    Find the nearest point in another geometry (line or polygon). More
    precisely,
    Source: https://gis.stackexchange.com/a/301935.

    Parameters
    ----------
    pts : geopandas.GeoDataFrame
        A dataframe containing the Point geometry. All columns on this dataframe
        are returned.
    df : geopandas.GeoDataFrame
        The dataframe
    df_cols : list[str]
        ...

    Returns
    -------
    geopandas.GeoDataFrame
        ...
    """
    A = np.concatenate(
        [np.array(geom.coords) for geom in pts.geometry.to_list()])
    B = [np.array(geom.coords) for geom in df.geometry.to_list()]
    B_ix = tuple(it.chain.from_iterable(
        [it.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = scipy.spatial.cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = op.itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [pts, df.loc[idx, df_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


def download_osm_db_shp(state=None, country='US', continent='North America',
                        outdir='', decompress=True):
    """
    Download the entire OSM database of the given region in the form of
    shapefiles and organize the layers into their folders.

    Parameters
    ----------
    state : str
        Name of the target province/state/national subdivision (optional).
    country : str
        Name of the target country.
    continent : str
        Name of the target continent.
    outdir : str
        Path of the folder where the OSM data will be downloaded and optionally
        organized.
    decompress : str
        Whether decompress the downloaded zipped file in the same folder and
        move the different shapefile components into their respective layers'
        folders.

    Returns
    -------
    fpath : str
        Path of the output file.
    headers : dict
        Response headers dict - contain information about the status of the
        API request's response.
    """
    domain = 'https://download.geofabrik.de'
    geos = [continent, country, state] if state else [continent, country]
    geos = '/'.join([x.lower().replace(' ', '-') for x in geos])
    url = f'{domain}/{geos}-latest-free.shp.zip'
    outdir = Path(outdir).mkdir(exist_ok=True, parents=True)
    out_path = outdir / 'osm.zip'
    fpath, headers = urllib.request.urlretrieve(url, str(out_path))
    if os.path.exists(fpath) and decompress:
        shutil.unpack_archive(fpath, outdir)
        for f in glob(outdir + '/gis_osm_*'):
            fname = f.split('/')[-1].replace('gis_osm_', '')
            layer = fname.split('_')[0]
            dir_ = (outdir / layer).mkdir(exist_ok=True)
            shutil.move(f, dir_ / fname)
    return fpath, headers
