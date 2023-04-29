import os, shutil
from pathlib import Path
import warnings

import pyspark
import pyspark.sql.functions as F

# Default configuration of the pyspark session. This dictionary is overwritten
# by the configuration parameters in the `project.yaml` file under the variable
# `spark_config`.
DEFAULT_CONFIG = {
    'sql.shuffle.partitions':               40,
    'driver.maxResultSize':                 0,
    'executor.memory':                      '36g',
    'executor.cores':                       10,
    'cores.max':                            10,
    'driver.memory':                        '36g',
    'default.parallelism':                  12,
    'sql.session.timeZone':                 'GMT',
    'sql.debug.maxToStringFields':          100,
    'sql.execution.arrow.pyspark.enabled':  'true',
}

class Types:
    """
    Aliases for common pyspark data types.
    """
    _ = pyspark.sql.types
    # simple types
    null = _.NullType()
    bool = _.BooleanType()
    int = _.IntegerType()
    int8 = _.ByteType()
    int16 = _.ShortType()
    int32 = _.IntegerType()
    int64 = _.LongType()
    float = _.FloatType()
    double = _.DoubleType()
    time = _.TimestampType()
    date = _.DateType()
    str = _.StringType()
    binary = _.BinaryType()
    # callable, composite types
    array = _.ArrayType
    map = _.MapType
    field = _.StructField
    schema = _.StructType


def schema(cols):
    """
    Build a pyspark schema with nullable fields from the given mapping of
    column names and data types.

    Parameters
    ----------
    cols : dict[str, <name in spark.py>]
        Columns keyed by column name. The value can either be a string
        exactly the same as one of the attribute names of class `T` or an
        instance of the attribute of class `T` directly. E.g., it can be
        either "int8" or `T.int8`.

    Returns
    -------
    pyspark.sql.types.StructField
        The desired pyspark schema object.
    """
    return Types.schema([Types.field(k, v, nullable=True)
                         for k, v in cols.items()])


def write(df, outdir, parts=None, compress=False, overwrite=True):
    """
    Save a pyspark dataframe as a parquet fle folder and overwrite the
    output directory if it already exists.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe to be written.
    outdir : str
        Folder to which parquet files will be written.
    parts : int | None
        Number of partitions (parquet files).
    compress : bool
        Whether compress the output files using `snappy` compression.
    overwrite : bool
        Whether overwrite an existing directory.
    """
    outdir = Path(outdir)
    if os.path.exists(outdir):
        if not overwrite:
            return f'Not overwriting existing directory: {outdir}'
        shutil.rmtree(outdir)
    if isinstance(parts, int):
        df = df.repartition(parts)
    outdir = outdir.parent.mkdir(parents=True, exist_ok=True) / outdir.stem
    if compress:
        (df.write.option('compression', 'none').mode('overwrite')
         .option('compression', 'snappy').save(outdir))
    else:
        df.write.parquet(outdir)


def zip_cols(df, key_cols, col_name):
    if isinstance(key_cols, str): key_cols = [key_cols]
    cols = [x for x in df.columns if not x in key_cols]
    return df.select(*key_cols, F.arrays_zip(*cols).alias(col_name))


# def unzip_cols(df, col_name='pts', uid=UID, x=LON, y=LAT, t=TS, e=ERR):
#     cols = [x, y, t, e] if e in str(df.schema) else [x, y, t]
#     if all([c in df.columns for c in cols]): return df
#     return df.select(uid, *[F.col(col)[x].alias(x) for x in cols])
        
        
class Spark:
    """
    A custom pyspark session handler to help with pyspark operations.
    """

    def __init__(self, config=None, log_level='WARN', start=False):
        """
        Parameters
        ----------
        config : dict[str, any]
            Custom configuration parameters in addition to the ones in
            `Spark.default_config`, by default listed in `project.yaml` ->
            `spark_config`.
        log_level : str
            Logging level for the project, taken from `logging` package.
        start : bool
            Whether start the pyspark session while constructing the object.
        """
        # take union of default config dictionary with given config
        config = DEFAULT_CONFIG | (config or {})
        self.config = {'spark.' + k: v for k, v in config.items()}
        self.log_level = log_level
        self.context = None
        self.session = None
        if start:
            self.start()

    def start(self):
        """
        Start pyspark session and store relevant objects.
        """
        if not self.context and not self.session:
            # set the configuration
            self.config = pyspark.SparkConf().setAll(list(self.config.items()))
            # create the context
            self.context = pyspark.SparkContext(conf=self.config)
            # start the session and set its log level
            self.session = pyspark.sql.SparkSession(self.context)
            self.session.sparkContext.setLogLevel(self.log_level)

    def empty_df(self, cols):
        """
        Create an empty dataframe with the given schema.

        Parameters
        ----------
        cols : dict[str, type]
            Mapping of column names with their target pyspark data types.

        Returns
        -------
        df : pyspark.sql.DataFrame
            Empty dataframe with the given schema.
        """
        schema = Types.schema([Types.field(k, v, nullable=True)
                               for k, v in cols.items()])
        df = self.context.emptyRDD()
        return self.session.createDataFrame(df, schema=schema)

    def read_csv(self, paths, schema=None, header=False):
        """
        Read CSV files and container folders as pyspark dataframes.

        Parameters
        ----------
        paths : str | list[str]
            Path(s) to one or more CSV file(s) or CSV-containing folder(s).
        schema : pyspark.sql.types.StructField
            Target schema of the dataframe.
        header : bool
            Whether read the first row of the file as columns.

        Returns
        -------
        pyspark.sql.DataFrame
            The loaded dataframe.
        """
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        df = (self.session.read.option('header', header)
              .csv(str(paths.pop(0)), schema))
        schema_ = df.schema
        for path in paths:
            df = df.union((self.session.read.option('header', header)
                           .csv(str(path), schema_)))
        return df

    def read_parquet(self, paths):
        """
        Read parquet files and container folders as pyspark dataframes.

        Parameters
        ----------
        paths : str | list[str]
            Path(s) to one or more parquet file(s) or parquet-containing
            folder(s).

        Returns
        -------
        pyspark.sql.DataFrame
            The loaded dataframe.
        """
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        return self.session.read.parquet(*[str(p) for p in paths])

    def pdf2sdf(self, df):
        """
        Convert a Pandas dataframe to a Pyspark dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.

        Returns
        -------
        df : pyspark.sql.DataFrame
            Output dataframe (same schema as the input dataframe).
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = self.session.createDataFrame(df)
        return df
