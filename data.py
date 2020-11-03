#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os


# #### OECD VERİSİ -  PWT


def oecd():
    data_oecd = pd.read_csv('data/DP_LIVE_19082020122503891.csv')

    data_oecd = data_oecd.rename(columns={'TIME': 'date'})

    # Frequency: Q (quarterly), Measure:  Index, Subject:  volidx Olanlar Seçiliyor
    data_oecd = data_oecd[(data_oecd['FREQUENCY'] == 'Q') & (
        data_oecd['MEASURE'] == 'IDX') & (data_oecd['SUBJECT'] == 'VOLIDX')]

    oecd = data_oecd.pivot(index='date', columns='LOCATION', values='Value')
    oecd.columns.name = None
    return oecd


def pwt():
    data_pwt = pd.read_excel(
        'data/pwt91.xlsx', sheet_name='Data', usecols=['year', 'countrycode', 'rgdpo'])

    data_pwt = data_pwt.rename(columns={'year': 'date'})

    pwt = data_pwt.pivot(
        index='date', columns='countrycode').droplevel(axis=1, level=0)
    pwt.columns.name = None
    return pwt


# #### EORA - WOID


def __read_eorawoid(path, rate_type, request_var, date=None):
    sector_data = pd.read_csv(path, index_col=['type', 'year', 'country'])
    if date is not None:
        sector_data = sector_data[sector_data.index.get_level_values(
            'year') >= date]

    sector_data = sector_data.sum(axis=1)

    sector_data = sector_data.groupby(
        level=['year', 'country'], group_keys=False).apply(lambda x: x / x.loc[rate_type])

    requested_data = sector_data.xs(request_var, level='type').sort_index()

    requested_data = requested_data.unstack()
    return requested_data


def from_eora(path='data/dataeora.csv', rate_type='gexp', request_var='gvc', date=None):
    return __read_eorawoid(path, rate_type, request_var, date)


def from_WOID(path='data/WOID_data.csv', rate_type='gexp', request_var='gvc', date=None):
    return __read_eorawoid(path, rate_type, request_var, date)


# #### Indicator


def indicator(name):
    if name == 'imf':
        return __imf_indicator()
    elif name == 'bl':
        indicator = pd.read_csv('data/X/lee&lee/indicator.csv')
    else:
        path = 'data/X/{}/indicator.csv'.format(name)
        indicator = pd.read_csv(path)
    return indicator


def __imf_indicator():
    available = pd.read_csv('data/X/imf/annually.csv',
                            usecols=['indicator'], squeeze=True).unique()

    imf_indicator = pd.read_excel('data/X/imf/indicator.xlsx', sheet_name='IFS',
                                  skiprows=1, usecols=['Indicator Name', 'Indicator Code'])

    imf_indicator = imf_indicator[imf_indicator['Indicator Code'].isin(
        available)]

    return imf_indicator


# #### IMF Data


def read_imf(code: str, frequency: str, date: int = None):
    base_path = 'data/X/imf/{}.csv'
    frequency = frequency.lower()
    if frequency == 'q':
        frequency = 'quarterly'
    elif frequency.lower() == 'a':
        frequency = 'annually'
    else:
        raise ValueError('frequency must be a (annualy) or q (quarterly)')

    path = base_path.format(frequency)
    data = pd.read_csv(path)
    if frequency == 'quarterly':
        data['date'] = data['date'].str.split().apply(
            lambda x: pd.Timestamp('-'.join([x[1], x[0]])))
    else:
        data['date'] = pd.to_datetime(data.date.astype(str))

    if date is not None:
        data = data[data.date >= str(date)]

    data = data[data.indicator == code].drop(
        'indicator', axis=1).set_index('date')

    return data


def __concat_excel():
    """To concatenate data in the willconcat folder.
    """

    base_path = 'data/X/willconcat'
    will_concat = list(map(lambda x: os.path.join(
        base_path, x), os.listdir(base_path)))
    df_list = []
    for path in will_concat:
        df_list.append(pd.read_excel(path, header=1))
    data = pd.concat(df_list)
    data = data.rename(
        columns={'Unnamed: 0': 'date', 'Unnamed: 1': 'indicator'})
    return data


# #### WB Data


def read_wb(code: str, date=None):
    base_path = 'data/X/wb/{}.csv'
    path = base_path.format(code)

    data = pd.read_csv(path)

    data.drop(data.tail(5).index, inplace=True)

    data.rename(columns={'Time Code': 'date',
                         'Series Code': 'indicator'}, inplace=True)

    data['date'] = data['date'].apply(
        lambda x: ''.join([ch for ch in x if ch.isdigit()]))

    data['date'] = pd.to_datetime(data.date.astype(str))

    if date is not None:
        data = data[data.date >= str(date)]

    data = data[data.indicator == code].drop(
        'indicator', axis=1).set_index('date')

    return data


# #### Lee & Lee


def __country_codes():
    codes = pd.read_html(
        'https://www.iban.com/country-codes')[0][['Country', 'Alpha-3 code']]

    some_countries = ['united kingdom', 'philippines', 'republic of korea', 'taiwan',
                      'czech republic', 'russian federation', 'dominican rep.',
                      'venezuela', 'iran', 'syria', 'congo, d.r.', 'cote divoire',
                      'gambia', 'niger', 'reunion', 'sudan', 'swaziland', 'netherlands', 'bolivia']
    some_codes = ['GBR', 'PHL', 'KOR', 'TWN', 'CZE', 'RUS', 'DOM', 'VEN', 'IRN',
                  'SYR', 'COD', 'CIV', 'GMB', 'NER', 'REU', 'SDN', 'SWZ', 'NLD', 'BOL']

    add = pd.DataFrame(list(zip(some_countries, some_codes)),
                       columns=['Country', 'Alpha-3 code'])

    codes = codes.append(add)

    codes.Country = codes.Country.str.lower()

    codes = codes.set_index('Country').unstack().droplevel(0)

    return codes


def __lee_hc(date: int = None):
    data = pd.read_excel('data/X/lee&lee/LeeLee_HC_MF1564 (1).xls', header=7)

    data = data.dropna(subset=['Year', 'Population\n(1000s)'])

    data.loc[:, 'Country'] = data['Country'].ffill().str.lower()

    data['Country'] = data['Country'].replace(__country_codes())

    data = data.rename(
        columns={'Age Group': 'Age Group 1', 'Unnamed: 3': 'Age Group 2'})

    data = data.astype({'Year': int, 'Age Group 1': int, 'Age Group 2': int})

    data = data.rename(columns={'Year': 'date'})

    return data


def __lee_enrol(date: int = None):
    data = pd.read_excel('data/X/lee&lee/LeeLee_enroll_MF (1).xls', header=7)

    data = data.dropna(subset=['Year'])

    data.loc[:, 'Country'] = data['Country'].ffill().str.lower()

    data['Country'] = data['Country'].replace(__country_codes())

    data = data.astype({'Year': int})

    data = data.rename(columns={'Year': 'date'})

    return data


def __lee_attain(date: int = None, ):
    data = pd.read_excel('data/X/lee&lee/LeeLee_attain_MF1564.xls',
                         header=7).rename(columns={'Unnamed: 3': 'Age Group 2'})

    data = data.dropna(how='all')

    data.loc[:, 'Country'] = data['Country'].ffill().str.lower()

    data['Country'] = data['Country'].replace(__country_codes())

    data = data.astype({'Year': int})

    data = data.rename(columns={'Year': 'date'})

    return data


def read_BL(code: str, variable: str, date: int = None):
    if code == 'hc':
        data = __lee_hc()
    elif code == 'enrol':
        data = __lee_enrol()
    elif code == 'attain':
        data = __lee_attain()
    else:
        raise ValueError(
            "Invalid code parameter. Please pass one of the parameters 'hc', 'enrol', 'attain'.")

    if date is not None:
        data = data[data['date'] >= date]

    data['Country'] = data['Country'].str.upper()

    data['date'] = pd.to_datetime(data.date.astype(str))

    data = data.pivot(index='date', columns='Country', values=variable)

    return data
