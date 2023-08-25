import warnings

from geopandas import GeoDataFrame, GeoSeries
import haversine as hs
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from shapely.geometry import Point
from sklearn.cluster import MeanShift

import mobilkit as mk
from mobilkit.spark import Types as T
from mobilkit.geo import CRS_DEG, CRS_M
from mobilkit.gps import UID, LAT, LON, TS


def get_stay_point_trips(x, y, t, dist_thresh, time_thresh):
    """
    Compute trip points based on the stay point detection algorithm in
    Sadeghinasr et al. (2019) https://doi.org/10.1061/9780784482438.002

    Parameters
    ----------
    x, y, t: list[float]
        Trajectory (sequence of pings), with x (longitude) and y (latitude)
        in degrees and t (time) in seconds since either the start of the day
        or the start of the first date in a set of dates.
    dist_thresh : float
        Maximum distance (in meters) between any two points in a stay region.
    time_thresh : float
        Maximum time gap (in seconds) between any two points in a stay region.

    Returns
    -------
    trips : list[list[tuple[3]]]
        A list of trips, each of which contains a sequence of points (x, y, t).
    """
    trips, cur_trip = [], []
    trip_started = False
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x):
            dist = float(hs.haversine((y[i], x[i]), (y[j], x[j]), unit='m'))
            tdiff = t[j] - t[i]
            if dist > dist_thresh:
                if tdiff > time_thresh:
                    coord = [sum(a[i:j])/(j-i) for a in [x, y]]
                    if trip_started:
                        cur_trip.append((*coord, t[i]))
                        trips.append(cur_trip)
                        cur_trip = []
                        trip_started = False
                    if not trip_started:
                        trip_started = True
                        cur_trip.append((*coord, t[j-1]))
                else:
                    if trip_started:
                        cur_trip.append((x[j], y[j], t[j]))
                i = j
                break
            j += 1
        if j == len(x):
            if trip_started:
                cur_trip.append((x[j-1], y[j-1], t[j-1]))
            break
    return trips


def segment_trips_clustering(row, min_dwell, radius, **ms_kwargs):
    """
    Segment trips from a given daily trajectory. This is done by first creating
    virtual cluster regions based on (x, y) coords that are later split by time.
    The clusters are labeled as 'stay' or 'enroute' depending on their dwell
    time. The pings between each pair of consecutive stay regions are grouped
    and categorized as one trip.

    Parameters
    ----------
    row : pandas.Series
        A dataframe row with at least the following columns:
        ---------------------------------------
        column  dtype       description
        ---------------------------------------
        $UID    int64       unique user ID
        $LON    [float]     longitudes of the pings
        $LAT    [float]     latitudes of the pings
        $TS     [float]     seconds between ping time & start of the day
    min_dwell : float
        Minimum time between two pings (seconds) needed to classify the time
        difference as "staying" at the same place.
    radius : float
        Kernel radius for MeanShift clustering (meters). This value is converted
        to degrees using `mobilkit.geo.dist_m2deg` based on a trajectory's mean
        latitude.
    ms_kwargs : dict
        Other parameters passed to the MeanShift algorithm.

    Returns
    -------
    trips : list[dict[str, int | float]]
        List of trips for the input user-day, each item being a dict of:
        $UID                int     input user ID
        trip_num            int     trip number for this user
        $LON, $LAT, $TS     [float] coordinates & timestamp of each point
    """
    # convert the trajectory's coordinates to long format and sort by time
    df = pd.DataFrame(row[[LON, LAT, TS]].to_dict()).sort_values(TS)
    # compute kernel bandwidth in degrees from the kernel radius
    bandwidth = mk.geo.dist_m2deg(radius, df[LAT].mean())
    try:
        # create and fit a MeanShift model with the (x, y) coordinate data
        model = MeanShift(bandwidth=bandwidth, **ms_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(df[[LON, LAT]])
            # extract the cluster labels and centers
            df['label'] = model.labels_
            df[['cx', 'cy']] = model.cluster_centers_[df['label']]
    except ValueError:
        # in case of any error, return an empty table with the expected schema
        return pd.DataFrame([], columns=['trip_num', UID, LON, LAT, TS])
    # redefine the clusters to make sure that temporally distant but
    # spatially close points are recognized as different clusters
    df['cid'] = (df['label'].diff() != 0).astype('int').cumsum()
    # get earliest & latest ping time of each redefined cluster
    C = df.groupby('cid')[TS].agg(['first', 'last'])
    # label the regions as stay/enroute based on their stay duration (seconds)
    C['enroute'] = C['last'] - C['first'] < min_dwell
    # join the cluster labels to the ping table
    df = df.merge(C['enroute'], on='cid')
    # collect ping points (x, y, t) for each cluster (row) in lists
    df['xyt'] = list(zip(df[LON], df[LAT], df[TS]))
    C = df.groupby(['cid', 'enroute'])['xyt'].agg(list).reset_index()
    # set same cluster IDs for consecutive enroute regions
    C['rgn_id'] = (((C['cid'] * (1 - C['enroute'].astype(int))).diff() != 0)
                   .astype(int).cumsum())
    # collect (combine) the points of consecutive enroute regions
    C = C.groupby(['rgn_id', 'enroute'])['xyt'].sum().reset_index()
    # identify trips & their ping coordinates from the stay/enroute region 
    # data using the logic used in the Maputo report (p.47)
    trip_started = True
    trip_num = 0
    X, trips = [], []
    # for each virtual region (row)
    for enroute, xyt in zip(C['enroute'], C['xyt']):
        if enroute:  # if this is a non-stay (enroute) region
            if not trip_started:
                # add all points of the enroute region to the trip
                X += xyt
        else:  # if this is a stay region
            # if region's last point starts a new trip
            if trip_started:
                trip_started = enroute
                # add the last point of this region to the trip
                X.append(xyt[-1])
            else:  # if this region's first point ends the current trip
                # add the first point of this region to the trip
                X.append(xyt[0])
                # collect the points for this trip
                trips.append({UID: row[UID], 'trip_num': trip_num} |
                             dict(zip([LON, LAT, TS], list(zip(*X)))))
                # create a new trip starting with this region's last point
                trip_num += 1
                X = [xyt[-1]]
    # return the segmented trips for this user-day
    return trips


def old_get_trip_summary(df, morn_peak_hrs, eve_peak_hrs):
    """
    Add summary details for each trip, including its travel time, total
    distance travelled (as sum of Haversine distances between the consecutive
    trip points), average trip speed, and the classification of the trip as in
    peak or off-peak period.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The output of `mobilkit.trips.get_trip_points` with the schema:
        ---------------------------------------
        column      dtype           description
        ---------------------------------------
        user_id     int64           user ID
        date        date            date (derived from timestamp)
        trip_id     int16           trip serial number per user-day
        lon         array[float]    longitudes of the trip points
        lat         array[float]    latitudes ""
        time        array[int32]    timestamps ""
        n_pts       int16           number of trip points
    morn_peak_hrs, eve_peak_hrs : tuple[float, float]
        2-tuples containing the range of hours defined as "morning peak hours"
        and "evening peak hours", e.g., morning -> (7, 10), evening -> (16, 19)

    Returns
    -------
    pyspark.sql.DataFrame
        A table with the following columns in addition to the input:
        -------------------------------
        column      dtype   description
        -------------------------------
        [[input]]           [[columns of input `df`]]
        time_min    float   trip travel time (in minutes)
        start_hr    float   hour when the trip started (range [0., 24.))
        end_hr      float   "" trip ended ""
        dist_m      float   total trip distance (in meters)
        speed_kmph  float   mean trip speed (in kmph)
        peak_period int8    numeric code denoting whether the trip is classified
                            as 'morning peak trip' (=1), 'off-peak trip' (=2),
                            or 'evening peak trip' (=3)
    """
    # travel time
    df = (df.withColumn('start_epoch', F.element_at('time', 1))
          .withColumn('end_epoch', F.element_at('time', -1))
          .withColumn('time_min', ((F.col('end_epoch') - F.col('start_epoch')) /
                                   60).cast(T.float))
          .withColumn('start_ts', F.from_unixtime('start_epoch'))
          .withColumn('end_ts', F.from_unixtime('end_epoch'))
          .withColumn('start_hr', (F.hour('start_ts') + F.minute('start_ts') /
                                   60).cast(T.float))
          .withColumn('end_hr', (F.hour('end_ts') + F.minute('end_ts') /
                                 60).cast(T.float))
          .drop('start_epoch', 'end_epoch', 'start_ts', 'end_ts'))

    # total distance
    def udf_dist(x, y):
        """ UDF to compute distance between two 1D coordinate arrays. """
        dist = 0
        start = (x[0], y[0])
        for i in range(1, len(x)):
            end = (x[i], y[i])
            dist += hs.haversine(start, end, unit=hs.Unit.METERS)
            start = end
        return dist

    df = df.withColumn('dist_m', F.udf(udf_dist, T.float)(df.lon, df.lat))
    # mean speed
    df = df.withColumn('speed_kmph', ((F.col('dist_m') / 1000) /
                                      (F.col('time_min') / 60)).cast(T.float))

    # peak period kind
    def udf_peak_period(start, end):
        morn_start, morn_end = morn_peak_hrs
        eve_start, eve_end = eve_peak_hrs
        if start < morn_start:  # started early morning
            if end < morn_start:  # ended before morning peak began
                code = 'off-peak'
            else:  # ended during or after morning peak
                code = 'morning peak'
        elif morn_start <= start <= morn_end:  # started during morning peak
            code = 'morning peak'
        elif morn_end < start < eve_start:  # started in afternoon off-peak
            if end < eve_start:  # ended before evening peak
                code = 'off-peak'
            else:  # ended during or after evening peak
                code = 'evening peak'
        elif eve_start <= start <= eve_end:  # started in evening peak
            code = 'evening peak'
        else:  # started after evening peak
            code = 'off-peak'
        codes = {'morning peak': 1, 'off-peak': 2, 'evening peak': 3}
        return codes[code]

    df = (df.withColumn('peak_period', F.udf(udf_peak_period, T.int8)(
        df.start_hr, df.end_hr)).drop('start_hr', 'end_hr'))
    return df


def old_get_trips(pings, min_slots, bandwidth, min_dwell, morn_peak_hrs,
                  eve_peak_hrs):
    """
    Parameters
    ----------
    pings : pyspark.sql.DataFrame
        Input pings table with at least the following columns:
        -------------------
        column        dtype
        -------------------
        user_id       int64
        lat           float
        lon           float
        timestamp     int32
        error_radius  float
        dt            str
        date          date
    min_slots : int

    bandwidth : float
        Bandwidth of RBF kernel to be used in MeanShift. For more details, see
        `mobilkit.gps.home_work_locs`.
    min_dwell : float
        Minimum time between two pings needed to classify the time difference
        as "staying" at the same place. Its unit is the same as that of the
        timestamp (usually seconds).
    morn_peak_hrs, eve_peak_hrs : tuple[float, float]
        2-tuples containing the range of hours defined as "morning peak hours"
        and "evening peak hours", e.g., morning -> (7, 10), evening -> (16, 19)

    Returns
    -------
    pyspark.sql.DataFrame
    """
    # get high-quality user-days
    user_day_stats = mk.gps.user_day_ping_counts(pings)
    hq_user_days = user_day_stats.filter(f'n_half_hrs >= {min_slots}')
    # filter pings with high-quality user-days
    pings = (pings.join(hq_user_days, on=['user_id', 'date'])
             .select('user_id', 'lat', 'lon', 'timestamp'))
    # create virtual regions (clusters)
    clustered_pings = old_make_virtual_regions(pings, bandwidth=bandwidth)
    # extract trip points from the clusters
    trip_pts = old_get_trip_points(clustered_pings, min_dwell)
    # add summary columns to the trip data
    trips = old_get_trip_summary(trip_pts, morn_peak_hrs, eve_peak_hrs)
    return trips


def old_segment_trips_coarse(df, min_pings, max_time):
    """
    Assign trip segment IDs from user pings based on threshold of sampling
    interval (i.e., time difference between consecutive pings) and the minimum
    ping count per segment.

    References: Paipuri et al. (2020) https://doi.org/10.1016/j.trc.2020.102709
    and Xu et al. (2020) https://doi.org/10.1140/epjds/s13688-021-00267-w

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Pings table of a given date with the schema generated by
        `mobilkit.gps.collect_pings_by_user`:
        ---------------------------------------
        column      dtype           description
        ---------------------------------------
        user_id     int64           user ID
        lat         array[float]    latidudes of the pings
        lon         array[float]    longitudes of the pings
        time_ms     array[int32]    time (ms) between ping & start of the day
    min_pings : int
        Minimum pings in each trip segment.
    max_time : float
        Minimum time difference (in seconds) between two consecutive pings
        required to split the segment.

    Returns
    -------
    df : pyspark.sql.DataFrame
        Table with the following schema:
        ---------------------------------------
        column      dtype           description
        ---------------------------------------
        user_id     int64           user ID
        seg_id      int             trip segment ID for this user
        lat         array[float]    latidudes of the pings
        lon         array[float]    longitudes of the pings
        time_ms     array[int32]    time (ms) between ping & start of the day
    """
    # remove low-frequency pings to improve speed
    df = df.withColumn('n_pings', F.size('time_ms'))
    df = df.filter(F.col('n_pings') >= min_pings)

    # cluster consecutive pings based on max sampling interval to get segments
    def udf(x): return [0] + np.cumsum(np.diff(x) >= max_time * 1000).tolist()

    df = df.withColumn('seg_id', F.udf(udf, T.array(T.int))('time_ms'))
    # explode all pings data to group it by trip segment
    cols = ['lon', 'lat', 'time_ms', 'seg_id']
    df = df.withColumn('xytc', F.explode(F.arrays_zip(*cols)))
    df = df.select('user_id', *[df['xytc'][x].alias(x) for x in cols])
    # collect ping data for each user-segment
    df = df.groupby('user_id', 'seg_id').agg(*[
        F.collect_list(x).alias(x) for x in cols[:-1]])
    # remove segments with few pings
    df = df.filter(F.size('time_ms') >= min_pings)
    return df


def old_segment_trips_fine(df, min_spd, max_spd):
    """
    Assign trip segment IDs from user pings based on threshold of sampling
    interval (i.e., time difference between consecutive pings) and the minimum
    ping count per segment.

    References: Paipuri et al. (2020) https://doi.org/10.1016/j.trc.2020.102709
    and Xu et al. (2021) https://doi.org/10.1140/epjds/s13688-021-00267-w

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Pings table of a given date with the schema generated by
        `mobilkit.gps.segment_trips_coarse`:
        ---------------------------------------
        column      dtype           description
        ---------------------------------------
        user_id     int64           user ID
        seg_id      int32           trip segment ID
        lat         array[float]    latidudes of the pings
        lon         array[float]    longitudes of the pings
        time_ms     array[int32]    time (ms) between ping & start of the day
    min_spd : float
        Minimum mean trip speed (m/s) needed to keep one trip intact.
    max_spd : float
        Maximum mean trip speed (m/s) to consider a trip physically valid.

    Returns
    -------
    df : pyspark.sql.DataFrame
        Table with the following schema:
        ---------------------------------------
        column      dtype           description
        ---------------------------------------
        user_id     int64           user ID
        seg_id      int             trip segment ID for this user
        lat         array[float]    latidudes of the pings
        lon         array[float]    longitudes of the pings
        time_ms     array[int32]    time (ms) between ping & start of the day
        time        array[float]    cumulative time difference (s) b/w pings
        dist        array[float]    cumulative distance (m) b/w pings
        mean_spd    float           mean trip speed (m/s)
    """

    # compute cumulative time difference b/w consecutive pings for each segment
    def udf(t):
        dt = np.diff(t) / 1000  # convert time from milliseconds to seconds
        if np.any(dt < 0):  # remove erroneous records with -ve time difference
            return None
        return np.cumsum(dt).tolist()

    # add the time difference column and remove erroneous (null) trip segments
    df = df.withColumn('time', F.udf(udf, T.array(T.float))('time_ms')).dropna()

    # compute cumulative distance b/w consecutive pings for each segment
    def udf(x, y, unit=hs.Unit.METERS):
        p1, p2 = list(zip(y[:-1], x[:-1])), list(zip(y[1:], x[1:]))
        return np.cumsum(hs.haversine_vector(p1, p2, unit)).tolist()

    # add the cumulative distance column
    df = df.withColumn('dist', F.udf(udf, T.array(T.float))('lon', 'lat'))

    # compute mean trip speed and remove those with < min speed
    def udf(d, t): return d[-1] / (t[-1] + 1e-6)

    df = df.withColumn('mean_spd', F.udf(udf, T.float)('dist', 'time'))
    df = df.filter(F.col('mean_spd').between(min_spd, max_spd))
    return df


def snap_trips(df, geom, snap_tol):
    """
    Snap a set of trip points to a given LineString-type layer.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Base trips table with the following schema:
        ---------------------------------------
        column      dtype           description
        ---------------------------------------
        user_id     int64           user ID
        trip_id     int             trip ID for this user
        lon, lat    array[float]    coordinates of the pings (in degrees, with
                                    the CRS being "EPSG:4326")
        ts          array[float]    time (s) between ping & start of the day
    geom : geopandas.GeoDataFrame
        The line geometry layer to which the trip points are to be snapped.
        For generality, the index of this table is reset for compatibility.
        The data here should be in the same CRS as of `df`.
    snap_tol : float
        Snapping tolerance, i.e., distance (meters) up to which a line is
        searched for each trip point. Normally, the coordinate units of both the
        dataframes are degrees, so an approximate value in degrees is used based
        on `mobilkit.geo.dist_m2deg()`.

    Returns
    -------
    df : pandas.DataFrame
        Snapped trip points along with their IDs for matching with the original
        input table `df`.
    """
    # assign trip point index
    df = df.withColumn('pt_id', F.udf(lambda x: list(range(len(x))), T.array(
        T.int))('lon').cast(T.array(T.float)))

    # convert snap tolerance from meters to degrees, depending on trips' points
    def udf(x):
        return mk.geo.dist_m2deg(snap_tol, np.mean(x))

    sdf = df.withColumn('tol', F.udf(udf, T.float)('lat'))
    # get the bounding box of each point, offset by the computed snap tolerance
    for new, old, op in [('minx', 'lon', -1), ('miny', 'lat', -1),
                         ('maxx', 'lon', 1), ('maxy', 'lat', 1)]:
        def udf(x, tol): return [y + op * tol for y in x]

        sdf = sdf.withColumn(new, F.udf(udf, T.array(T.float))(old, 'tol'))
    # zip together all the point-wise info
    cols = ['pt_id', 'lon', 'lat', 'ts', 'minx', 'miny', 'maxx', 'maxy']
    df = sdf.select(F.arrays_zip(*cols).alias('pt'))
    # convert to pandas dataframe since the GIS task of snapping cannot be done
    # in pyspark (as of writing: 2022-12-12)
    # assign a unique trip id, explode per point & split point info into columns
    df = df.toPandas().rename_axis('tr_id').explode('pt').reset_index()
    for c, val in zip(cols, list(zip(*df.pop('pt')))): df[c] = val
    # make sure the line layer's units are degrees & is properly indexed
    geom = geom.to_crs(CRS_DEG).reset_index(drop=True).rename_axis('line_id')
    # for each point, identify the line IDs which intersect its offset bound box
    df = df.set_index(['tr_id', 'pt_id', 'lon', 'lat', 'ts'])
    df = (df.apply(lambda r: list(geom.sindex.intersection(
        r[['minx', 'miny', 'maxx', 'maxy']])), axis=1)
          .rename('line_id').explode().reset_index().dropna()
          .astype({'tr_id': np.int32, 'pt_id': np.int16, 'line_id': np.int32}))
    # generate point geometry & project to spatial CRS
    df['pt_geom'] = GeoSeries([Point(x) for x in zip(
        df['lon'], df['lat'])], crs=CRS_DEG).to_crs(CRS_M)
    # join the line layer (projected to spatial CRS) with the points layer
    df = (geom.to_crs(CRS_M)['geometry'].rename('line_geom').reset_index()
          .merge(df, on='line_id'))
    # compute the distance (meters) between each point-line pair
    df['snap_dist'] = df['line_geom'].distance(df['pt_geom'])
    # remove the point-line pairs which exceed the input snap tolerance
    df = df[df['snap_dist'] <= snap_tol].sort_values('snap_dist')
    # for each point, select only the closest line feature
    df = df.groupby(['tr_id', 'pt_id']).first().reset_index()
    # for each point, find its distance with the closest point on its line
    df['d_pt2cp'] = df['line_geom'].project(df['pt_geom'])
    # find the closest ("snapped") point along the line
    df['geometry'] = df['line_geom'].interpolate(df['d_pt2cp'])
    # create a table with the snapped point geometry & get the new coordinates
    df = GeoDataFrame(df, geometry=df['geometry'], crs=CRS_M).to_crs(CRS_DEG)
    df = pd.concat([df.drop(columns=['lon', 'lat']),
                    mk.geo.gdf2pdf(df, 'lon', 'lat')], axis=1)
    # for each trip, collect the point-wise attributes into lists
    df = (df.groupby('tr_id')[['pt_id', 'lon', 'lat', 'ts', 'line_id']]
          .agg(list).reset_index())
    # map the created trip ID column `tr_id` back to each user's trip number
    df2 = sdf.select('user_id', 'trip_id').toPandas()
    df = df2.merge(df, left_index=True, right_on='tr_id').drop(columns='tr_id')
    return df

