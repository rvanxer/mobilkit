"""
Only applicable for US Census and American Community Survey (ACS) analysis.
"""
from itertools import zip_longest
import re
import requests

import pandas as pd
from pandas import DataFrame as Pdf

# base URL for the census data API
ROOT_URL = 'https://api.census.gov/data'

# Census table types, obtained from
# https://www.census.gov/programs-surveys/acs/data/data-tables/table-ids
# -explained.html
TABLE_TYPES = Pdf([  # columns: (type id, label, description)
    ('B', 'Detailed Tables - Base Table',
     'Most detailed estimates on all topics for all geographies'),
    ('C', 'Collapsed Table',
     'Similar information from its corresponding Base Table (B) but at a '
     'lower level of detail because one or more lines in the Base Table have '
     'been grouped together'),
    ('S', 'Subject Table',
     'A span of information on a particular ACS subject, such as veterans, '
     'presented in the format of both estimates and percentages'),
    ('R', 'Ranking Table',
     'State rankings across approximately 90 key variables'),
    ('GCT', 'Geographic Comparison Table',
     'Comparisons across approximately 95 key variables for geographies other '
     'than states such as counties or congressional districts'),
    ('K20', 'Supplemental Table',
     'Simplified Detailed Tables at a lower population threshold than the '
     'standard 1-year tables (all areas with populations of 20,000 or more)'),
    ('XK', 'Experimental Estimates',
     'Experimental estimates, which are different from the standard ACS data '
     'releases'),
    ('DP', 'Data Profile',
     'Broad social, economic, housing, and demographic information in a total '
     'of four profiles'),
    ('NP', 'Narrative Profile',
     'Summaries of information in the Data Profiles using nontechnical text'),
    ('CP', 'Comparison Profile',
     'Comparisons of ACS estimates over time in the same layout as the Data '
     'Profiles'),
    ('S0201', 'Selected Population Profile',
     'Broad ACS statistics for population subgroups by race, ethnicity, '
     'ancestry, tribal affiliation, and place of birth'),
], columns=('id', 'label', 'info'))

# Census table subjects, obtained from
# https://www.census.gov/programs-surveys/acs/data/data-tables/table-ids
# -explained.html
SUBJECTS = { # subject_id: subject_label
    '01': 'Age; Sex',
    '02': 'Race',
    '03': 'Hispanic or Latino Origin',
    '04': 'Ancestry',
    '05': 'Citizenship Status; Year of Entry; Foreign Born Place of Birth',
    '06': 'Place of Birth',
    '07': 'Migration/Residence 1 Year Ago',
    '08': 'Commuting (Journey to Work); Place of Work',
    '09': 'Relationship to Householder',
    '10': 'Grandparents and Grandchildren Characteristics',
    '11': 'Household Type; Family Type; Subfamilies',
    '12': 'Marital Status; Marital History',
    '13': 'Fertility',
    '14': 'School Enrollment',
    '15': 'Educational Attainment; Undergraduate Field of Degree',
    '16': 'Language Spoken at Home',
    '17': 'Poverty Status',
    '18': 'Disability Status',
    '19': 'Income',
    '20': 'Earnings',
    '21': 'Veteran Status; Period of Military Service',
    '22': 'Food Stamps/Supplemental Nutrition Assistance Program (SNAP)',
    '23': 'Employment Status; Work Status Last Year',
    '24': 'Industry, Occupation, and Class of Worker',
    '25': 'Housing Characteristics',
    '26': 'Group Quarters',
    '27': 'Health Insurance Coverage',
    '28': 'Computer and Internet Use',
    '29': 'Citizen Voting-Age Population',
    '98': 'Quality Measures',
    '99': 'Allocation Table for Any Subject'
}

# Important ACS5 (2020) fields inspected manually in the Excel file downloaded
# from https://www2.census.gov/programs-surveys/acs/tech_docs/table_shells/table_lists/2020_DataProductList_5Year.xlsx
# This list is not constant - entries can be added & removed as per requirement.
IMP_FIELDS = {  # {field label: field ID in the 'Detailed Tables'}
    # overall
    'B01001_001E': 'popu', # total population
    'B19001_001E': 'hh', # total no. of households
    # age
    'B16004_002E': 'age_minor', # population of minors (aged 5-17y)
    'B16004_024E': 'age_adult', # population of adults (aged 18-64y)
    'B16004_046E': 'age_senior', # population of seniors (aged ≥65y)
    'B01002_001E': 'age_median', # median age of the entire population
    # sex
    'B01001_026E': 'sex_female', # total female population
    'B01001_002E': 'sex_male', # total male population
    # race
    'B02001_002E': 'race_white', # total White population
    'B02001_003E': 'race_black', # total Black population
    # education
    'B15003_001E': 'edu_eligible', # total population aged ≥25y
    'B15011_001E': 'edu_bachelors', # num. people having Bachelor's degree or higher
    # employment
    'B24080_001E': 'employ_total', # total employed people (aged ≥16y)
    'B23025_002E': 'employ_in_LF', # population in labor force (LF)
    'B23025_007E': 'employ_not_in_LF',
    # income (in past 1 year; age ≥16y only)
    'B19313_001E': 'inc_total', # aggregate income
    'B19301_001E': 'inc_avg', # per capita income
    'B19025_001E': 'inc_total_hh', # aggregate household income
    'B19013_001E': 'inc_median', # overall median houshehold income
    # poverty (based on income in past 1 year; age ≥16 only)
    'B17007_002E': 'pop_poor', # population below poverty line
    'B17007_023E': 'pop_nonpoor', # "" aboe poverty line ""
    'B17017_002E': 'hh_poor', # num. households below poverty line ""
    'B17017_031E': 'hh_nonpoor', # num. households above poverty line ""
    # commute to work (no. of workers who used this mode)
    'B08301_003E': 'cm_car', # car (drove alone)
    'B08301_004E': 'cm_pool', # carpooling
    'B08301_010E': 'cm_pt', # public transit
    'B08301_011E': 'cm_bus', # bus
    'B08301_012E': 'cm_subway', # subway
    'B08301_016E': 'cm_taxi', # taxicab
    'B08301_018E': 'cm_bike', # bicycle
    'B08301_019E': 'cm_walk', # walking
    'B08301_002E': 'cm_wfm', # work from home
}


def get_subdivs(geo, src, year, key=None):
    """
    Get the list of political subdivisions and their FIPS codes for US regions
    using the US Census/ACS API.

    Parameters
    ----------
    geo : list[tuple[str, str]]
        Geography specification of the region(s) of interest.
        Examples:
            >> [('state', '18'), ('county', '*')] # all counties in Indiana
            >> [('state', '18'), ('tract', '*')] # all tracts in Indiana
    src : str
        Data source: either decennial census summary table ('sf1') or one of
        ACS datasets ('acs1', 'acs3', or 'acs5')
    year : int
        Year of the data.
    key : str
        Census data API key to be registered by each user.

    Returns
    -------
    pandas.DataFrame | requests.Response
        If the request is successful, this function should return a table
        containing the names of the political subdivisions along with their
        FIPS codes.
        Example:
            >> name                     state    county
            >> ----                     -----    ------
            >> White County, Indiana       18       181
            >> ...
    """
    assert src in ['acs1', 'acs3', 'acs5', 'sf1'], f'Incorrect `src`: {src}'
    geo_ = [(x[0].replace(' ', '+'), x[1]) for x in geo]
    for_ = '&for=' + ':'.join(geo_.pop(-1))
    in_ = '&in=' + '+'.join(':'.join(x) for x in geo_)
    params = 'get=NAME' + for_ + (in_ if len(geo_) > 0 else '')
    params += (f'&key={key}' if isinstance(key, str) else '')
    pre_src = 'acs' if 'acs' in src else 'dec'
    url = f'{ROOT_URL}/{year}/{pre_src}/{src}?{params}'
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            cols = [x.lower() for x in data.pop(0)]
            return pd.DataFrame(data[1:], columns=cols)
    return resp


def get_fields(year, src, table_type='detail'):
    """
    Fetch the ACS dataset description using the US Census API. This is mainly
    used to get the description of the variables in the different ACS tables.

    Parameters
    ----------
    year : int
        Year of the dataset.
    src : str
        Source of the dataset: must be one of {'acs1', 'acs5', 'acsse'}.
    table_type : str
        Table type as defined by the US Census. Allowed values are the 1 or
        2-character codes as defined in the `id` column of `mk.acs.TABLE_TYPES`.

    Returns
    -------
    pandas.DataFrame
        A table containing the ACS fields with their details. Columns:
            id              Unique field ID.
            label           Description of the field.
            concept         Label of the table containing this field.
            predicateType   Data type of the output (predicate)
            group           ID of the table containing this field.
            limit           ???
            attributes      Other related fields to the estimate, e.g., margin
                            of error.
            required        ???
    """
    table_type = '/' + table_type if table_type != 'detail' else ''
    url = f'{ROOT_URL}/{year}/acs/{src}{table_type}/variables.json'
    resp = requests.get(url)
    if resp.status_code == 200:
        vars_ = list(resp.json()['variables'].items())
        fields = Pdf(dict(vars_[3:])).T.rename_axis('id').reset_index()
        return fields
    elif resp.status_code == 404:
        raise ValueError('Error 404: ' + url)
    else:
        raise ValueError('Non-404 error: ' + url)


def process_fields(orig_fields):
    """
    Clean the ACS fields table obtained from `mk.acs.fetch_fields()` so that
    the information of that table is easy to read and use.

    Parameters
    ----------
    orig_fields : pandas.DataFrame
        The result table of `mk.acs.fetch_fields()`.

    Returns
    -------
    pandas.DataFrame
        Cleaned table with the following columns (at least for ACS5 2020 data):
            type            Table type, defined in `mk.acs.TABLE_TYPES`.
            subject_id      2-digit character code of the table subject.
            subject         Label of the table subject.
            table_id        Unique table ID, e.g., 'B02001A'.
            number          The table number within a particular subject.
            table           Name/label of the table.
            race_id         Character code of the race identifier, if any.
            race_label      Name of the race identifier in the table, if any.
            is_PR           Whether this table is for Puerto Rico.
            field_id        Unique field ID, e.g., 'B02001A_002E'.
            field           Name/label of the field, separated in scope by `__`.
            dtype           Data type of the field output (predicate).
            L0, L1,...,L7   Columns representing the hierarchy of the field
                            name, obtained as `field.split('__')`.
    """
    # Fields
    fields = (
        orig_fields.drop(columns=['concept', 'limit', 'required', 'attributes'])
        .rename(columns={'predicateType': 'dtype', 'group': 'table_id'})
        .query('label != "Geography"')
        .assign(label=lambda df: [x.replace('!!', '__').replace(':', '')
                                  for x in df['label']]))
    expanded = (Pdf(list(zip(*zip_longest(
        *fields['label'].apply(lambda x: x.split('__')).tolist(), fillvalue=''
    ))), index=fields['id']).rename(columns=lambda x: f'L{x}').reset_index())
    fields = (fields.merge(expanded, on='id').sort_values('id')
              .rename(columns={'id': 'field_id', 'label': 'field'}))

    # Tables
    tab = (orig_fields.groupby(['group', 'concept'])
           .size().reset_index().drop(columns=0)
           .rename(columns={'group': 'table_id', 'concept': 'table'}))
    tab['table'] = tab['table'].str.title()
    tab['type'] = tab['table_id'].str.slice(0, 1)
    tab['subject'] = tab['table_id'].str.slice(1, 3)
    tab['number'] = tab['table_id'].str.slice(3, 6)
    tab['is_PR'] = tab['table_id'].str.endswith('PR')
    tab['race_id'] = [re.sub(r'\d', '', x[-1]) for x in
                      tab['table_id'].replace(r'PR$', '', regex=True)]
    tab['race_label'] = tab['table'].apply(
        lambda x: [y.strip() for y in x.replace(')', '').split('(')][-1])
    tab.loc[tab['race_id'] == '', 'race_label'] = ''
    tab.loc[tab['race_id'] != '', 'table'] = tab['table'].apply(
        lambda x: x.split('(')[0].strip())
    tab = tab.rename(columns={'subject': 'subject_id'})[
        ['table_id', 'type', 'subject_id', 'number', 'race_id',
         'is_PR', 'table', 'race_label']]

    # combine the subject, censustable, and field tables into one table
    subj = (pd.Series(SUBJECTS).rename('subject').rename_axis('subject_id')
            .reset_index())
    # subj = SUBJECTS.rename(columns=dict(id='subject_id', label='subject'))
    acs = (tab.merge(subj, on='subject_id')
           .merge(fields, on='table_id')
           [['type', 'subject_id', 'subject', 'table_id', 'number', 'table',
             'race_id', 'race_label', 'is_PR', 'field_id', 'field', 'dtype'] +
            list(fields.filter(regex='^L').columns)]
           .applymap(lambda x: x.lower() if type(x) == str else x))
    return acs


def search_field(fields, **params):
    """
    Search for the ID of a particular ACS field in the processed `fields` table
    using different search terms.

    Parameters
    ----------
    fields : pandas.DataFrame
        Result of `mk.acs.process_fields()`.
    params : dict


    Returns
    -------
    str | pandas.DataFrame
        Either the ID of the desired field (if one search result is found) or a
        table containing all the approximate matches.
    """
    func = params.pop('func', None)
    if 'race_id' not in params:
        params.update({'race_id': ''})
    params = list({k: v.lower() for k, v in params.items()}.items())
    query = ' and '.join(['{} == "{}"'.format(*x) for x in params])
    res = fields.query(query)
    if func is not None:
        res = res.pipe(func)
    if res.shape[0] == 1:
        return res.iloc[0]['field_id'].upper()
    return res


def download(geo, fields, src='acs5', year=2020, table_type='detail', key=None):
    """
    Download the data of the given ACS fields for a given region.

    Parameters
    ----------
    geo : list[tuple[str, str]]
        Regional geography defined as a list of subregions (e.g., a state is
        made of counties). The geographical hierarchy is to be defined as a list
        or tuple of 2-tuples: (geographical scale, search criteria). Here, the
        search criterion can be the FIPS code of the region or wildcard. E.g.,
        geo = [('state', '18'), ('county', '*')] for all counties in Indiana.
        For more details, see the API document:
        https://census.gov/content/dam/Census/data/developers/api-user-guide/api-guide.pdf
    fields : list[str]
        List of codes for the data fields of interest. E.g., ['B01001E',
        'C15027PR', ...]
    src : str
        Source of the dataset: must be one of {'acs1', 'acs5', 'acsse'}.
    year : int
        Year of the dataset.
    table_type : str
        Table type as defined by the US Census. Allowed values are the 1 or
        2-character codes as defined in the `id` column of `mk.acs.TABLE_TYPES`.
    key : str
        Census Data API key (optional but highly recommended).
    Returns
    -------
    pandas.DataFrame
        Table with the rows denoting the spatial units, the index denoting their
        identity hierarchy, and the columns named the same as :attr:`fields` and
        having the data values.
    """
    assert src in ['acs1', 'acs5', 'acsse', 'sf1']
    assert table_type in ['detail', 'profile', 'subject']
    geo_ = [(x[0].replace(' ', '+'), x[1]) for x in geo]
    for_ = '&for=' + ':'.join(geo_.pop(-1))
    in_ = '&in=' + '+'.join(':'.join(x) for x in geo_)
    key = f'&key={key}' if isinstance(key, str) else ''
    chunksize = 49
    res = Pdf()
    for i in range(0, len(fields), chunksize):
        cols = fields[i : (i+chunksize)]
        get = 'get=' + ','.join(cols)
        url = f'{ROOT_URL}/{year}/acs/{src}?{get}{for_}{in_}{key}'
        resp = requests.get(url)
        try:
            data = resp.json()
            df = Pdf(data[1:], columns=data[0])
            id_cols = list(set(df.columns) - set(cols))
            res = pd.concat([res, df.set_index(id_cols)], axis=1)
        except Exception as e:
            print('Failed fetching', cols)
            print(e)
    return res.astype(float)
