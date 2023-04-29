# from mobilkit import (dt, F, hs, np, Path, pyspark, pytz, reduce)

import datetime as dt
from functools import reduce
import os
from pathlib import Path
import warnings

# from haversine import haversine_vector as haversine
import haversine as hs
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as Sdf
import pytz
import sklearn
# from sklearn.cluster import MeanShift

from mobilkit.spark import Types as T
from mobilkit.spark import write
from mobilkit.utils import mkdir

# column names for Quadrant data
UID = 'uid'  # unique user identifier
LON = 'lon'  # longitude (degrees)
LAT = 'lat'  # latitude (degrees)
TS = 'ts'  # timestamp or time since beginning of day (usually in seconds)
ERR = 'error'  # GPS error radius (meters), an indicator of ping's accuracy


def read_cols(df, cols=[
    ('_c0', UID, T.str),
    ('_c3', LON, T.float),
    ('_c2', LAT, T.float),
    ('_c5', TS, T.int64),
    ('_c4', ERR, T.float)
]):
    """
    Read specific columns of a raw/original Quadrant table.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Pings table in the original/raw Quadrant CSV.gz data files.
    cols : list[tuple[str, str, pyspark.sql.types.<Type>]]
        Columns of interest, given as 3-tuples: (original name, new name, )
        - `old`: original column name, usually '_c0', '_c1', etc.
        - `new`: new desired column name (should ideally be templated)
        - `dtype`: desired data type just after reading the data.

    Returns
    -------
    df : pyspark.sql.DataFrame
        Table with the selected columns, cast into the input data types.
    """
    return df.select([F.col(old).cast(dtype).alias(new) for old, new, dtype in cols])


def get_user_pings(sp, date, inroot, outroot=None, tz='UTC',
                   in_fmt='year={}/month={}/day={}', out_fmt='%Y/%m/%d'):
    """
    Collect the ping coordinates and timestamps for each user (in a row) within
    one orignal (raw) Quadrant ping data file/folder after adjusting for time
    difference.

    Parameters
    ----------
    sp : mobilkit.spark.Spark
        Pyspark session handler.
    date : Date
        Date of interest.
    inroot : str
        Folder or file containing raw/original pings in .gz.csv format. Their
        columns are not named.
    outroot : optional[str]
        If it is given, write the data to its subdirectory named after the date.
    tz : str
        Name of time zone to which the pings have to be shifted. E.g.,
        "US/Eastern".
    new_method : bool
        If true, the data was collected using the 'T-date' method, where a
        date's data folder includes all pings received in the Quadrant system
        on the date, irrespective of their timestamps.
        If false, the data was collected using the 'P-date' method, where a
        date's data folder only includes the pings with timestamps
        corresponding to that date.
    window : int
        When `new_method == True`, this value represents the no. of days of
        which a given date's data folder contains the data. Typically, this is
        7 days, including one day of the present date and 6 days before that.
    in_fmt : str
        String format of the given date's data folder inside the `inroot`
        directory. E.g., format = 'year={}/month={}/day={}' for a folder called
        '{inroot}/year=2022/month=03/day=01'.
    out_fmt : str
        String format of the given date for the output data subfolder's name.
        E.g., '%Y-%m-%d' will result in a folder '{outroot}/2022-03-01'.

    Returns
    -------
    df : pyspark.sql.DataFrame
        Table with the following schema:
        ---------------------------------------
        column  dtype       description
        ---------------------------------------
        $UID    int64       numeric user IDs converted from the original
                            alphanumeric device IDs
        $LON    [float]     longitudes of the pings
        $LAT    [float]     latitudes of the pings
        $TS     [float]     seconds between ping time & start of the day
        $ERR    [float]     GPS error radii of the pings (meters)
    """
    # resolve the directory containing the data of a given date `d`
    def get_path(d): return Path(inroot) / in_fmt.format(*str(d).split('-'))
    # read the data of the given date
    df = sp.read_csv(get_path(date))
    # select only relevant columns, rename them & change their dtypes
    df = read_cols(df)
    # compute the timestamp of the start time of the given date in GMT
    start_gmt = dt.datetime.fromisoformat(str(date)).timestamp()
    # compute the time difference (in seconds) of the given time zone with GMT
    offset = (dt.datetime.fromisoformat(str(date)).astimezone(pytz.timezone(tz))
              .utcoffset().total_seconds())
    # get the time difference of the pings from the timestamp of the start of
    # the given date measured in given time zone
    df = df.withColumn(TS, F.col(TS) / 1000 - start_gmt)
    # filter the pings lying in the local time of [0, 24 hrs]
    df = df.filter(F.col(TS).between(0, 24*3600))
    # if the offset is negative (i.e., time zone west of GMT), read the
    # remainder of the pings present in the file of the next date
    if offset < 0:
        # if the file of the next date exists, read its data
        date2 = date + dt.timedelta(days=1)
        if get_path(date2).exists():
            df2 = read_cols(sp.read_csv(get_path(date2)))
            # get the time difference of the pings from the GMT timestamp of
            # the next day's start, shifted to local (given) time zone
            df2 = df2.withColumn(
                TS, F.col(TS) / 1000 - start_gmt - 24 * 3600)
            # filter the pings with negative time difference (i.e., the pings
            # recorded before 00:00 local time of next day)
            df2 = df2.filter(F.col(TS) < 0)
            # convert this time difference of next day to that of current day
            df2 = df2.withColumn(TS, F.col(TS) + 24 * 3600)
            # append this data to the main day's data
            df = df.union(df2)
    # if the offset is positive, read the remainder pings from previous date
    if offset > 0:
        # if the file of the previous date exists, read its data
        date2 = date - dt.timedelta(days=1)
        if get_path(date2).exists():
            df2 = read_cols(sp.read_csv(get_path(date2)))
            # get the time difference of the pings from the GMT timestamp of
            # the current day's start, shifted to local (given) time zone
            df2 = df2.withColumn(TS, F.col(TS) / 1000 - start_gmt)
            # filter the pings with positive time difference (i.e., the pings
            # recorded after 00:00 local time of current day)
            df2 = df2.filter(F.col(TS) > 0)
            # append this data to the main day's data
            df = df.union(df2)
    df = df.withColumn(TS, F.col(TS).cast(T.float))
    # convert string user IDs to integer-type to save space
    df = df.withColumn(UID, F.xxhash64(UID))
    # sort pings by time and then collect their coordinates for each user
    df = df.sort(TS).groupby(UID).agg(*[
        F.collect_list(x).alias(x) for x in [LON, LAT, TS, ERR]])
    # write the data
    if isinstance(outroot, str):
        write(df, mkdir(outroot) / date.strftime(out_fmt), compress=True)
    return df


def filter_bbox_long(df, left, bottom, right, top, xcol=LON, ycol=LAT):
    return df.filter(f'{left} <= {xcol} and {xcol} <= {right} and ' +
                     f'{bottom} <= {ycol} and {ycol} <= {top}')


def filter_bbox_zipped(df, left, bottom, right, top, col='pings', dtype=T.float):
    def udf(pings):
        return [tuple(p) for p in pings if
                (bottom <= p[0] <= top) and (left <= p[1] <= right)]
    df = df.withColumn(col, F.udf(udf, T.array(T.array(dtype)))(col))
    return df.filter(F.size(col) > 0)


def get_tdiff_zipped(df, time=TS, tdiff='tdiff', dtype=T.float):
    def udf(t):
        return [0.] + [float(t[i+1] - t[i]) for i in range(len(t)-1)]
    return df.withColumn(tdiff, F.udf(udf, T.array(dtype))(time))


def get_dist_zipped(df, x=LON, y=LAT, dist='dist', unit='m', dtype=T.float):
    def udf(x, y):
        try:
            src, trg = list(zip(y[:-1], x[:-1])), list(zip(y[1:], x[1:]))
            return [0.] + hs.haversine_vector(src, trg, unit=unit).tolist()
        except Exception:
            return []
    df = df.withColumn(dist, F.udf(udf, T.array(dtype))(x, y))
    return df.filter(F.size(dist) > 0)


def get_speed_zipped(df, dist='dist', tdiff='tdiff', speed='speed',
              zero_div_tol=1e-4, dtype=T.float):
    def udf(d, t):
        return [float(d / (t + zero_div_tol)) for d, t in zip(d, t)]
    return df.withColumn(speed, F.udf(udf, T.array(dtype))(dist, tdiff))


def get_accel_zipped(df, speed='speed', tdiff='tdiff', accel='accel',
              zero_div_tol=1e-4, dtype=T.float):
    def udf(v, t):
        return [0.] + [float((v[i+1] - v[i]) / (t[i+1] + zero_div_tol))
                       for i in range(1, len(v) - 1)]
    return df.withColumn(accel, F.udf(udf, T.array(dtype))(speed, tdiff))


def get_motion_metrics(df, zero_tol=1e-6, dist='dist', tdiff='tdiff', 
                       speed='speed', accel='accel', lon=LON, lat=LAT, ts=TS):
    """
    """
    df = df.filter(F.size(lon) > 2)
    def get_dist(x, y):
        src, trg = list(zip(y[:-1], x[:-1])), list(zip(y[1:], x[1:]))
        return [0.] + hs.haversine_vector(src, trg, unit='m').tolist()
    def get_tdiff(t):
        return [0.] + np.diff(t).astype(float).tolist()
    def get_speed(d, t):
        return (np.array(d) / (np.array(t) + zero_tol)).tolist()
    def get_accel(v, t):
        return [0., 0.] + (np.diff(v[1:]) / (np.array(t[2:]) + zero_tol)).tolist()
    df = df.withColumn(dist, F.udf(get_dist, T.array(T.float))(lon, lat))
    df = df.withColumn(tdiff, F.udf(get_tdiff, T.array(T.float))(ts))
    df = df.withColumn(speed, F.udf(get_speed, T.array(T.float))(dist, tdiff))
    df = df.withColumn(accel, F.udf(get_accel, T.array(T.float))(speed, tdiff))
    return df


def collect_days_data(sp, root, dates, fmt='%Y-%m-%d'):
    """
    Collect multiple days' user-ping data into one list by addting 
    the number of seconds of each day to the timestamp.
    """
    df = []
    dates = sorted(dates)
    for date in dates:
        d = sp.read_parquet(root / date.strftime(fmt))
        nDays = (date - dates[0]).days
        def add_day(t): return [t + nDays * 86400 for t in t]
        d = d.withColumn(TS, F.udf(add_day, T.array(T.float))(TS))
        df.append(d)
    df = reduce(pyspark.sql.DataFrame.union, df)
    df = df.groupby(UID).agg(*[F.flatten(F.collect_list(x)).alias(x) 
                               for x in [LON, LAT, TS]])
    return df


# def filter_user_pings_bbox(df, bbox, uid=UID, x=LON, y=LAT, t=TS, xyt='xyt',
#                            in_zipped=False, out_zipped=False):
#     """
#     Filter pings within a given bounding box.
#
#     Parameters
#     ----------
#     df : pyspark.sql.DataFrame
#         Input user pings table either with the schema of the output of
#         `mobilkit.gps.get_user_pings` or zipped records of the following schema,
#         depending on `:attr:in_zipped`:
#         ---------------------------------------
#         column  dtype       description
#         ---------------------------------------
#         $UID    int64       unique user ids
#         $xyt    [[float]]   list of ping records each containing a 3-tuple of
#                             (longitude, latitude, timestamp) for each ping
#     bbox : tuple/list[float]
#         Bounding box in the standard format: (minx, miny, maxx, maxy).
#     uid, x, y, t : str
#         Names of the user ID ($uid) column as well as the longitude ($x),
#         latitude ($y), and the time/timestamp ($t) columns in the input and
#         output tables.
#     xyt : str
#         Name of the column containing zipped ping records.
#     in_zipped, out_zipped : bool
#         Whether the input and output tables have zipped ping records. If yes,
#         there is no need to zip the records and is thus slightly faster.
#
#     Returns
#     -------
#     df : pyspark.sql.DataFrame
#         Input user pings table either with the schema of the output of
#         `mobilkit.gps.get_user_pings` or zipped records of the following schema,
#         depending on `:attr:out_zipped`:
#         ---------------------------------------
#         column  dtype       description
#         ---------------------------------------
#         $UID    int64       unique user ids
#         $xyt    [[float]]   list of ping records each containing a 3-tuple of
#                             (longitude, latitude, timestamp) for each ping
#     """
#     assert len(bbox) == 4, 'Bounding box must have length of 4'
#     minx, miny, maxx, maxy = bbox
#     if not in_zipped:
#         df = df.select(uid, F.arrays_zip(x, y, t).alias(xyt))
#     df = df.select(uid, F.udf(lambda xyt: list(zip(*[
#         p[:] for p in xyt if minx <= p[0] <= maxx and miny <= p[1] <= maxy
#     ])), T.array(T.array(T.float)))(xyt).alias(xyt))
#     if not out_zipped:
#         df = df.select(uid, *[F.col(xyt)[i].alias(c) for i, c in
#                               enumerate([x, y, t])])
#     return df


# # noinspection PyChainedComparisons
# def get_n_filter_motion_metrics(df, max_speed=np.inf, max_accel=np.inf,
#                                 max_decel=np.inf, div0_tol=1e-8, sort=True):
#     """
#     Compute the main metrics associated with a set of user ping trajectories
#     and filter unrealistic pings. These attributes are: (i) time since previous
#     ping, (ii) distance from previous ping, (iii) speed, and (iv) acceleration.
#
#     Parameters
#     ----------
#     df : pyspark.sql.DataFrame
#         Table with each row having the trajectory (set of (x, y, t)) of a user
#         as in the output of `mobilkit.gps.get_user_pings`:
#         ---------------------------------------
#         column  dtype       description
#         ---------------------------------------
#         $UID    int64       numeric user IDs converted from the original
#                             alphanumeric device IDs
#         $LON    [float]     longitudes of the pings
#         $LAT    [float]     latitudes of the pings
#         $TS     [float]     seconds between ping time & start of the day
#         ...     ...         other columns
#     div0_tol : float
#         A small number added to the denominator to avoid division by zero error
#         in speed and acceleration computation.
#     max_speed : float
#         Maximum allowed speed of a ping's segment (unit: m/s).
#     max_accel, max_decel : float
#         Maximum allowed acceleration and deceleration values (unit: m/s^2).
#     sort : bool
#         Whether the pings need to be sorted. Set False when it is already known
#         that the pings are already sorted to save computation time.
#
#     Returns
#     -------
#     df : pyspark.sql.DataFrame
#         Table with the following schema:
#         ---------------------------------------
#         column  dtype           description
#         ---------------------------------------
#         $UID    int64       numeric user IDs converted from the original
#                             alphanumeric device IDs
#         $LON    [float]     longitudes of the pings
#         $LAT    [float]     latitudes of the pings
#         $TS     [float]     seconds between ping time & start of the day
#         tdiff   [float]     time difference (s) b/w consecutive pings
#         dist    [float]     distance (m) b/w consecutive pings
#         speed   [float]     speed (m/s) of the ping segment
#         accel   [float]     acceleration (m/s^2) of the ping segment
#     """
#     # remove users who have < 3 pings (min needed to compute acceleration)
#     df = df.filter(F.size(LON) > 2)
#     # zip ping info (and sort the pings if needed)
#     def zip_sort(x, y, t):
#         xyt = list(zip(x, y, t))
#         if sort: xyt = sorted(xyt, key=lambda x: x[2])
#         return xyt
#     df = df.select(UID, F.udf(zip_sort, T.array(T.array(T.float)))
#                    (LON, LAT, TS).alias('xyt'))
#     # compute segment time (s)
#     def time(xyt): return [0.] + np.diff([p[2] for p in xyt]).tolist()
#     df = df.withColumn('t', F.udf(time, T.array(T.float))('xyt'))
#     # segment length/distance (m)
#     def dist(xyt):
#         def yx(xy): return [(p[1], p[0]) for p in xy]
#         return [0.] + haversine(yx(xyt[:-1]), yx(xyt[1:]), unit='m').tolist()
#     df = df.withColumn('d', F.udf(dist, T.array(T.float))('xyt'))
#     # segment speed (m/s)
#     def speed(d, t): return [float(d / (t + div0_tol)) for d, t in zip(d, t)]
#     df = df.withColumn('v', F.udf(speed, T.array(T.float))('d', 't'))
#     # segment acceleration (m/s^2)
#     df = df.withColumn('a', F.udf(lambda v, t: [0., 0.] + (np.diff(v) / (
#             np.array(t[1:]) + div0_tol)).tolist(), T.array(T.float))('v', 't'))
#     # zip together pings with their time, speed & acceleration info
#     df = df.select(UID, 'xyt', F.arrays_zip('t', 'd', 'v', 'a').alias('tdva'))
#     df = df.withColumn('X', F.udf(lambda x, y: [(*x, *y) for x, y in zip(x, y)],
#                                   T.array(T.array(T.float)))('xyt', 'tdva'))
#     # filter pings whose Δt > 0 and speed & acceleration are within bounds
#     df = df.withColumn('X', F.udf(lambda x: list(zip(*[
#         p for p in x if p[3] > 0  # segment time is non-zero
#         and p[4] <= max_speed and -max_decel <= p[5] <= max_accel
#     ])), T.array(T.array(T.float)))('X'))
#     # remove users with zero filtered ping
#     df = df.filter(F.size('X') > 0)
#     # unzip the columns
#     cols = enumerate([LON, LAT, TS, 'tdiff', 'dist', 'speed', 'accel'])
#     df = df.select(UID, *[F.expr(f'X[{i}]').alias(x) for i, x in cols])
#     return df
#
#
# def get_n_filter_motion_metrics_pd(df, max_speed=np.inf, max_accel=np.inf,
#                                    max_decel=np.inf, div0_tol=1e-8, sort=True):
#     """
#     Same as `mobilkit.gps.get_n_filter_motion_metrics` but using Pandas.
#
#     Parameters
#     ----------
#     * Same as `mobilkit.gps.get_n_filter_motion_metrics` except the input table
#     `df` being a Pandas dataframe instead of a Pyspark dataframe.
#
#     Returns
#     -------
#     * Same as `mobilkit.gps.get_n_filter_motion_metrics` except it is a Pandas
#     dataframe instead of a Pyspark dataframe.
#     """
#     # remove users who have < 3 pings (min needed to compute acceleration)
#     df = df[df[LON].apply(len) > 2]
#     # zip ping info
#     df = (df.set_index(UID).apply(lambda r: list(
#         zip(r[LON], r[LAT], r[TS])), axis=1).rename('X').reset_index())
#     # sort the pings if needed
#     if sort:
#         df['X'] = df['X'].apply(lambda x: sorted(x, key=lambda x: x[2]))
#     # compute segment time (s)
#     def time(x): return [0.] + np.diff([p[2] for p in x]).tolist()
#     df['tdiff'] = df['X'].apply(time)
#     # segment length/distance (m)
#     def dist(xyt):
#         def yx(xy): return [(p[1], p[0]) for p in xy]
#         return [0.] + haversine(yx(xyt[:-1]), yx(xyt[1:]), unit='m').tolist()
#     df['dist'] = df['X'].apply(dist)
#     # segment speed (m/s)
#     df['speed'] = df.apply(lambda r: [d / (t + div0_tol) for d, t in
#                                       zip(r['dist'], r['tdiff'])], axis=1)
#     # segment acceleration (m/s^2)
#     df['accel'] = df.apply(lambda r: [0., 0.] + (np.diff(r['speed']) / (
#         np.array(r['tdiff'][1:]) + div0_tol)).tolist(), axis=1)
#     # zip together pings with their time, speed & acceleration info
#     df = df.set_index(UID).apply(lambda r: [
#         (*xyt, t, d, v, a) for xyt, t, d, v, a in zip(
#             *[r[x] for x in ['X', 'tdiff', 'dist', 'speed', 'accel']])
#     ], axis=1)
#     # filter pings whose Δt > 0 and speed & acceleration are within bounds
#     df = df.apply(lambda x: [
#         p for p in x if p[3] > 0 and p[4] <= max_speed and
#         -max_decel <= p[5] <= max_accel])
#     # remove users with zero filtered ping
#     df = df[df.apply(len) > 0].rename('X')
#     # unzip the columns
#     df = pd.concat([
#         df.apply(lambda x: [p[i] for p in x]).rename(c) for i, c in
#         enumerate([LON, LAT, TS, 'tdiff', 'dist', 'speed', 'accel'])
#     ], axis=1).reset_index()
#     return df


# def filter_hq_pings(sp, paths, bbox=None, min_pings=1, max_error=np.inf,
#                     max_speed=np.inf, max_accel=np.inf,
#                     max_decel=np.inf, div0_tol=1e-4):
#     """
#     Filter high-quality (HQ) events (pings) from saved ping data based on
#     certain criteria. Currently, these criteria include:
#         * Only pings lying in the given bounding box are allowed.
#         * Only pings with geospatial accuracy more than a threshold are allowed.
#         * The speed of a ping's segment (i.e., distance from previous ping /
#             time difference from previous ping) must be upper bounded.
#         * The acceleration of a ping's segment pair (i.e., the speed difference
#             between a ping's segment speed from previous segment's speed / time
#             difference this ping and two pings before) should be upper and
#             lower bounded.
#         * After filtering pings based on the above criteria, only users with
#             ping count more than a given minimum are filtered.
#
#     Parameters
#     ----------
#         sp : mobilkit.spark.Session
#             Pyspark session handler.
#         paths : list(str)
#             List of clean ping data folders (each having a date-formatted
#             path of parquet files: "{root}/{YYYY}/{MM}/{DD}").
#         bbox : tuple[float, 4]
#             Bounding box given by (xmin, ymin, xmax, ymax) (in degrees).
#         max_error : float
#             Maximum allowed GPS error radius of a ping (unit: m).
#         max_speed : float
#             Maximum allowed speed of a ping's segment (unit: m/s).
#         max_accel, max_decel : float
#             Maximum allowed acceleration and deceleration values (unit: m/s^2).
#         min_pings : int
#             Minimum no. of pings for a user-day to be considered valid.
#         div0_tol : float
#             A small number added to the denominator to avoid division by zero
#             error in speed and acceleration computation.
#
#     Returns
#     -------
#         df : pyspark.sql.DataFrame
#             Table with each row containing record array of a user:
#             column  dtype      description
#             ------  -----      -----------
#             $UID    int64      unique user ID
#             $LON    [float]    longitudes of the pings
#             $LAT    [float]    latitudes ""
#             $TS     [float]    seconds between ping & start of the day
#     """
#     # read the data
#     df = sp.read_parquet(paths)
#     # zip info of each record
#     df = df.select(UID, F.arrays_zip(LON, LAT, TS, ERR).alias('X'))
#     # at least remove the definitely low-frequency users to speed up processing
#     df = df.filter(F.size('X') >= min_pings)
#     # user defined function template for ping records
#     def udf(f, col): return F.udf(f, T.array(T.array(T.float)))(col).alias(col)
#     # remove erroneous pings (allow NaN)
#     df = df.select(UID, udf(lambda x: [p[:3] for p in x if p[3] == np.nan
#                                        or p[3] <= max_error], 'X'))
#     # filter pings within bounding box
#     if bbox and len(bbox) == 4:
#         minx, miny, maxx, maxy = bbox
#         df = df.select(UID, udf(lambda x: [
#             p for p in x if minx <= p[0] <= maxx and miny <= p[1] <= maxy], 'X'))
#     # remove users who have few filtered pings (and at least 3 pings needed to
#     # compute acceleration)
#     df = df.filter(F.size('X') >= max(3, min_pings))
#     # sort pings by time
#     df = df.select(UID, udf(lambda x: sorted(x, key=lambda k: k[2]), 'X'))
#     # segment time (s)
#     def time(x): return [0.] + [x[i+1][2] - x[i][2] for i in range(len(x) - 1)]
#     df = df.withColumn('t', F.udf(time, T.array(T.float))('X'))
#     # segment length/distance (m)
#     df = df.withColumn('d', F.udf(lambda x: [0.] + haversine(*[
#         [(z[1], z[0]) for z in y] for y in [x[:-1], x[1:]]], unit='m')
#                                      .tolist(), T.array(T.float))('X'))
#     # segment speed (m/s)
#     def speed(d, t): return [float(d / (t + div0_tol)) for d, t in zip(d, t)]
#     df = df.withColumn('v', F.udf(speed, T.array(T.float))('d', 't'))
#     # segment acceleration (m/s^2)
#     df = df.withColumn('a', F.udf(lambda v, t: [0., 0.] + (np.diff(v) / (
#             np.array(t[1:]) + div0_tol)).tolist(), T.array(T.float))('v', 't'))
#     # zip together pings with their time, speed & acceleration info
#     df = df.select(UID, 'X', F.arrays_zip('t', 'd', 'v', 'a').alias('Y'))
#     df = df.select(UID, F.udf(lambda x, y: [(*x, *y) for x, y in zip(x, y)],
#                               T.array(T.array(T.float)))('X', 'Y').alias('X'))
#     # remove pings whose Δt > 0 and speed & accel are within bounds and retain
#     # only their (x, y, t) coordinates
#     df = df.withColumn('X', F.udf(lambda x: list(zip(*[
#         p[:3] for p in x if p[3] > 0 # segment time is non-zero
#         and p[4] <= max_speed # speed is upper bounded
#         and -abs(max_decel) <= p[5] <= max_accel # acceleration is bounded
#     ])), T.array(T.array(T.float)))('X'))
#     # remove users with too few filtered pings
#     df = df.filter(F.size('X') >= min_pings)
#     # unzip the columns
#     df = df.select(UID, *[F.expr(f'X[{i}]').alias(x) for i, x in
#                           enumerate([LON, LAT, TS])])
#     return df


# def user_day_ping_counts(df):
#     """
#     Compute the aggregate statistics related to the number of pings per user for
#     a specific (unknown) date. These stats may be helpful later to filter
#     high-quality users or user-days.
#
#     Parameters
#     ----------
#     df : pyspark.sql.DataFrame
#         Pings table of a given (unknown) date with the schema generated by
#         `mobilkit.gps.collect_pings_by_user`:
#         ---------------------------------------
#         column      dtype           description
#         ---------------------------------------
#         user_id     int64           user ID
#         lat         array[float]    latidudes of the pings
#         lon         array[float]    longitudes ""
#         time_ms     array[int32]    milliseconds between ping & start of the day
#
#     Returns
#     -------
#     pyspark.sql.DataFrame
#         Table containing the following columns per user:
#         -------------------------------
#         column      dtype   description
#         -------------------------------
#         user_id     int64   user ID
#         n_pings     int32   total no. of pings of this user on this day
#         n_half_hrs  int8    no. of unique half-hour slots covered by the pings
#         n_hrs       int8    no. of unique hours covered by the pings
#     """
#     df = df.withColumn('n_pings', F.size(LAT))
#
#     def udf1(arr): return [t // (0.5 * 3600) for t in arr]
#     df = df.withColumn('half_hr', F.udf(udf1, T.array(T.float))(TS))
#
#     def udf2(arr): return [t // (3600 * 1000.) for t in arr]
#     df = df.withColumn('hour', F.udf(udf2, T.array(T.float))(TS))
#     df = df.withColumn('n_half_hrs', F.size(
#         F.array_distinct('half_hr')).cast(T.int8))
#     df = df.withColumn('n_hrs', F.size(F.array_distinct('hour')).cast(T.int8))
#     df = df.select(UID, 'n_pings', 'n_half_hrs', 'n_hrs')
#     return df


def get_home_work_loc_data(sp, dates, root, day_hrs, min_pings=None,
                               kind='home', path_date_fmt='%Y-%m-%d'):
    """
    Prepare the ping data to be used to estimate users' home or work locations.
    This function collects all the relevant pings in the target hours (night
    hours for home and day hours for work locations) over a given list of dates,
    accounting for boundary date conditions. If provided, it also removes users
    with too few pings overall. The output of this function can be used in
    different home/work location estimation algorithms.

    Parameters
    ----------
    sp : mobilkit.spark.Spark
        Pyspark session handler.
    dates : listlike[datetime.date]
        List of dates for which the ping data are to be collected.
    root : str
        Base directory containing the daily ping data folders.
    day_hrs : tuple[float, 2]
        Starting and ending time of the "daytime" period (in hours). E.g., a
        value of (8, 21) means that the daytime is defined as 8 AM through 9 PM.
        Equivalently, the nighttime is 9 PM through 8 AM.
    min_pings : int
        Minimum no. of pings needed per user overall. Too few pings mean that
        the data is not reliable for a user, so it is wise to remove such users.
    kind : str
        Either "home" or "work". The target period of each day is decided based
        on this value along with the `:attr:day_hrs` parameter. If its value is
        "home", then this function excludes the morning period of the first date
        and includes the morning period of the date next to the last date.
    path_date_fmt : str
        Date string format of the ping data folders inside `:attrs:root`.

    Returns
    -------
    df : pyspark.sql.DataFrame
        Table containing the (x, y) coordinates of each user in each row:
        ---------------------------
        column  dtype   description
        ---------------------------
        $UID    int64   user ID
        $LON    [float] total no. of pings of this user on this day
        $LAT    [float] no. of unique half-hour slots covered by the pings
    """
    assert kind in ['home', 'work']
    # convert the start and end times of the daytime from hours to seconds
    start, end = [x * 3600 for x in day_hrs]
    # template for resolving the ping data directory for a given date
    def get_dir(date): return Path(root) / date.strftime(path_date_fmt)
    # read the data of a date
    def read_date(date): return (sp.read_parquet(get_dir(date))
                                 .withColumn('date', F.lit(date)))
    # template for zipping the (x, y, t) coordinates of pings
    def zip_xyt(df):
        df = df.select('date', UID, F.arrays_zip(LON, LAT, TS).alias('xyt'))
        return df.filter(F.size('xyt') > 0)
    # template for filtering pings ((x, y, t)) with a boolean mask function
    def filt_xy(df, mask):
        return df.withColumn('xyt', F.udf(lambda xyt: list(zip(*[
            (x, y, t) for x, y, t in xyt if mask(t)
        ])), T.array(T.array(T.float)))('xyt'))
    # read the data of the first date
    df = zip_xyt(read_date(dates[0]))
    if kind == 'home':
        # for home locations, select only the evening time of the first day
        df = filt_xy(df, lambda t: t >= end)
        # for the remaining days, filter pings during both morning & evening
        for date in dates[1:]:
            df2 = zip_xyt(read_date(date))
            df = df.union(filt_xy(df2, lambda t: t <= start or t >= end))
        # for the last day, select only the morning time (if its data exists)
        next_day_dir = get_dir(dates[-1] + dt.timedelta(days=1))
        if next_day_dir.exists():
            df2 = zip_xyt(read_date(dates[-1] + dt.timedelta(days=1)))
            df = df.union(filt_xy(df2, lambda t: t <= start))
    elif kind == 'work':
        # for work locations, simply select all the pings during daytime
        for date in dates[1:]:
            df = df.union(zip_xyt(read_date(date)))
        df = filt_xy(df, lambda t: start <= t <= end)
    df = df.filter(F.size('xyt') > 0)
    # get day difference from the 1st day
    df = df.withColumn('day', (F.col('date') - dates[0]).cast(T.int16))
    # filter users who have sufficient records during this period
    if isinstance(min_pings, int):
        df2 = df.select(UID, F.size('xyt').alias('n'))
        df2 = df2.groupby(UID).agg(F.sum('n').alias('n_pings'))
        df2 = df2.filter(f'n_pings >= {min_pings}').select(UID)
        df = df.join(df2, on=UID)
    # split coordinates into two columns
    df = df.select(UID, 'day', *[F.col('xyt')[i].alias(x) for i, x in
                                 enumerate([LON, LAT, TS])])
    return df


# def _meanshift_home_work_locs(data: np.array, bandwidth: float, outliers: bool):
#     """
#     Get coordinates of the largest cluster's center for home/work location
#     using Scikit Learn's MeanShift clustering implementation
#     (https://scikit-learn.org/stable/modules/generated/sklearn.cluster
#     .MeanShift.html).
#     Input shape: (num_samples, num_features=2)
#     Output shape: (1, num_features=2)
#     """
#     try:
#         model = MeanShift(bandwidth=bandwidth, bin_seeding=True,
#                           cluster_all=outliers)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             model.fit(data)
#         return model.cluster_centers_[0].tolist()
#     except ValueError:
#         return [np.nan, np.nan]
#
#
# def get_home_work_locs(df, bw, day_hours, func=_meanshift_home_work_locs,
#                        kind='home', outliers=True):
#     """
#     Get the home or work location of each user using the ping data during the
#     given day/night hours. This involves clustering the GPS points using
#     meanshift clustering.
#
#     Parameters
#     ----------
#     df : pyspark.sql.DataFrame
#         Pings table of a given (unknown) date with the schema generated by
#         `mobilkit.gps.collect_pings_by_user`:
#         ---------------------------------------
#         column      dtype           description
#         ---------------------------------------
#         user_id     int64           user ID
#         lat         array[float]    latidudes of the pings
#         lon         array[float]    longitudes ""
#         time_ms     array[int32]    time (ms) between ping & start of the day
#     bw : float
#         Bandwidth of RBF kernel to be used in MeanShift. If the lat and lon are
#         in degrees, this value needs to be in degrees, not regular distance
#         units. A reasonable approximation can be found using meter-to-degree
#         conversion in `mobilkit.geo.dist_m2deg` for which the approximate mean
#         latitude of the target place needs to be given.
#         CAUTION: If this parameter is not given, the algorithm can take
#         sufficiently more time as described in the documentation.
#     func : function
#         Wrapper function used to do clustering for each user. This argument is
#         only added because of a pickling issue in the implementation in Pyspark.
#     kind : str
#         Kind of place to be inferred. Allowed values are 'home' and 'work'.
#         'home' uses nightime hours and 'work' uses daytime hours given by
#         `day_hours`.
#     day_hours : tuple[float, float]
#         Starting and ending hours in the range of daytime.
#     outliers : bool
#         If true, include orphan points (those that did not get any cluster with
#         the given bandwidth) in their closest clusters. If there are any major
#         outliers, they can substantially influence the means of the clusters,
#         so the decision for this flag is to be judicious. See the documentation
#         for details.
#
#     Returns
#     -------
#     pyspark.sql.DataFrame
#         Processed dataframe with the following schema.
#         -------------------------------
#         column      dtype   description
#         -------------------------------
#         user_id     int64   unique user ID (different from original device ID)
#         n_pings     int32   total no. of pings per user
#         lon, lat    float   coordinates of the home/work location
#     """
#     # zip the lat, lon, and time arrays into one for processing
#     df = df.withColumn('xyt', F.arrays_zip(LON, LAT, 'time_ms'))
#     # convert the daytime period boundaries from hours to milliseconds
#     low, high = [x * 3600 * 1000 for x in day_hours]
#     # filter pings which lie in the required time range (within `day_hours` if
#     # work locations are estimated or outside `day_hours` for home locations)
#     df = df.withColumn('coords', F.udf(
#         lambda arr: [(x, y) for x, y, t in arr if
#                      (low <= t <= high) == (kind == 'work')],
#         T.array(T.array(T.float)))('xyt'))
#     # remove users who have no pings after filtering in the required time period
#     df = df.withColumn('n_pings', F.size('coords')).filter(F.col('n_pings') > 0)
#     # do the clustering to obtain the coordinates of the home/work location
#     df = df.withColumn('center', F.udf(
#         lambda x: func(x, bandwidth=bw, outliers=outliers),
#         T.array(T.float))('coords'))
#     return df.select('user_id', 'n_pings', df.center[0].alias(LON),
#                      df.center[1].alias(LAT))
