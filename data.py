#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import pandas.tseries.offsets as offset

# #### OECD VERİSİ -  PWT

datasets = ['imf', 'bl', 'pwt', 'wb', 'eora', 'woid']


def oecd(frequency: str, measure: str, subject: str):
    data_oecd = pd.read_csv('data/DP_LIVE_19082020122503891.csv')

    data_oecd = data_oecd.rename(columns={'TIME': 'date'})

    # Frequency: Q (quarterly), Measure:  Index, Subject:  volidx Olanlar Seçiliyor
    data_oecd = data_oecd[(data_oecd['FREQUENCY'] == frequency.upper()) & (
            data_oecd['MEASURE'] == measure) & (data_oecd['SUBJECT'] == subject)]

    oecd = data_oecd.pivot(index='date', columns='LOCATION', values='Value')
    oecd.columns.name = None
    return oecd


def read_pwt(code: str, date: int = None):
    """
    PWT veri setinden belirli verileri okuyun.

    Args:
        code: İstenen indicator'a ait kod.
        date: Default None. Veri hangi yıldan itibaren alınsın?
    """
    data = pd.read_excel(
        'data/pwt91.xlsx', sheet_name='Data', usecols=['year', 'countrycode', code])

    data = data.rename(columns={'year': 'date'})

    data['date'] = pd.to_datetime(data.date.astype(str))

    if date is not None:
        data = data[data.date >= str(date)]

    data = data.pivot(index='date', columns='countrycode').droplevel(axis=1, level=0)

    data.columns.name = None

    return data


def read_eora(code: str, date=None):
    """
    Seçilen değişkene ait sektörlerin toplamını döndürür.

    Args:
        code: İstenen indicator'a ait kod.
        date: Default None. Veri hangi yıldan itibaren alınsın?
    """
    data = pd.read_csv('data/X/eora/eora.csv')
    data = data.rename(columns={'year': 'date'})
    data['date'] = pd.to_datetime(data.date.astype(str))
    data = data[data.type == code]
    if date is not None:
        data = data[data.date >= str(date)]

    data = data \
        .set_index(['date', 'country']) \
        .sum(axis=1) \
        .reset_index() \
        .pivot(index='date', columns='country', values=0)

    return data


def read_woid(code: str, date=None):
    """
    Seçilen değişkene ait sektörlerin toplamını döndürür.

    Args:
        code: İstenen indicator'a ait kod.
        date: Default None. Veri hangi yıldan itibaren alınsın?
    """
    data = pd.read_csv('data/X/woid/WOID_data.csv')
    data = data.rename(columns={'year': 'date'})
    data['date'] = pd.to_datetime(data.date.astype(str))
    data = data[data.type == code]
    if date is not None:
        data = data[data.date >= str(date)]

    data = data \
        .set_index(['date', 'country']) \
        .sum(axis=1) \
        .reset_index() \
        .pivot(index='date', columns='country', values=0)

    return data


# #### source


def source(name: str = None) -> pd.DataFrame:
    """
    Veri setlerine ait indicator ve kodları almak için kullanılır.
    Parametre geçilmezse veri setlerini belirten mesaj döndürür.

    Args:
        name (str) : Default None. İstenen veri seti:
            data.datesets ile belirtilmiştir.
    """
    if name is None:
        return "{} veri setlerinden birini kullanın".format(datasets)
    if name == 'imf':
        return __imf_source()
    elif name == 'bl':
        definition = pd.read_csv('data/X/lee&lee/indicator.csv')
    elif name == 'pwt':
        return __pwt_source()
    elif name == 'wb':
        definition = pd.read_csv('data/X/wb/indicator.csv')
    elif name == 'eora':
        definition = pd.read_csv('data/X/eora/indicator.csv')
    elif name == 'woid':
        definition = pd.read_csv('data/X/woid/indicator.csv')
    elif name == 'oecd':
        return __source_oecd()
    else:
        raise ValueError('No source with specified name.')
    return definition


def __imf_source():
    available = pd.read_csv('data/X/imf/annually.csv',
                            usecols=['indicator'], squeeze=True).unique()

    imf_source = pd.read_excel('data/X/imf/indicator.xlsx', sheet_name='IFS',
                               skiprows=1, usecols=['Indicator Name', 'Indicator Code'])

    imf_source = imf_source[imf_source['Indicator Code'].isin(
        available)]

    return imf_source


def __source_oecd():
    data_oecd = pd.read_csv('data/DP_LIVE_19082020122503891.csv')
    return dict(measure=data_oecd['MEASURE'].unique().tolist(), subject=data_oecd['SUBJECT'].unique().tolist())


def __pwt_source():
    return pd.read_excel('data/pwt91.xlsx', sheet_name='indicator')


# #### IMF Data


def read_imf(code: str, frequency: str, date: int = None):
    """
    IMF veri setinden belirli verileri okuyun.

    Args:
        code: İstenen indicator'a ait kod.
        frequency (str) : quarterly veriler kontrol edilmek isteniyorsa 'q',
            annually veriler için 'a' ayarlanmalı.
        date: Default None. Veri hangi yıldan itibaren alınsın?
    """
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

    data = data[~data.index.duplicated()]

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
    """
    World Bank veri setinden belirli verileri okuyun.

    Args:
        code: İstenen indicator'a ait kod.
        date: Default None. Veri hangi yıldan itibaren alınsın?
    """
    base_path = 'data/X/wb/{}.csv'
    path = base_path.format(code)

    data = pd.read_csv(path)

    data = data.drop(data.tail(5).index).dropna(how='all')

    data.rename(columns={'Time Code': 'date',
                         'Series Code': 'indicator'}, inplace=True)

    data['date'] = data['date'].apply(
        lambda x: ''.join([ch for ch in x if ch.isdigit()]))

    data['date'] = pd.to_datetime(data.date.astype(str))

    if date is not None:
        data = data[data.date >= str(date)]

    data = data.drop('indicator', axis=1).set_index('date')

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


def __lee_hc():
    data = pd.read_excel('data/X/lee&lee/LeeLee_HC_MF1564 (1).xls', header=7)

    data = data.dropna(subset=['Year', 'Population\n(1000s)'])

    data.loc[:, 'Country'] = data['Country'].ffill().str.lower()

    data['Country'] = data['Country'].replace(__country_codes())

    data = data.rename(
        columns={'Age Group': 'Age Group 1', 'Unnamed: 3': 'Age Group 2'})

    data = data.astype({'Year': int, 'Age Group 1': int, 'Age Group 2': int})

    data = data.rename(columns={'Year': 'date'})

    return data


def __lee_enrol():
    data = pd.read_excel('data/X/lee&lee/LeeLee_enroll_MF (1).xls', header=7)

    data = data.dropna(subset=['Year'])

    data.loc[:, 'Country'] = data['Country'].ffill().str.lower()

    data['Country'] = data['Country'].replace(__country_codes())

    data = data.astype({'Year': int})

    data = data.rename(columns={'Year': 'date'})

    return data


def __lee_attain():
    data = pd.read_excel('data/X/lee&lee/LeeLee_attain_MF1564.xls',
                         header=7).rename(columns={'Unnamed: 3': 'Age Group 2'})

    data = data.dropna(how='all')

    data.loc[:, 'Country'] = data['Country'].ffill().str.lower()

    data['Country'] = data['Country'].replace(__country_codes())

    data = data.astype({'Year': int})

    data = data.rename(columns={'Year': 'date'})

    return data


def read_bl(code: str, date: int = None):
    """
    Barro-Lee Verisini okuyun.

    Args:
        code: (str) : İstenen indicator'a karşılık gelen kod.
        date: (int) : Veri hangi yıldan itibaren okunsun?

    """
    file = code.partition('_')[0]
    code = code.partition('_')[-1]
    if file == 'hc':
        data = __lee_hc()
    elif file == 'enrol':
        data = __lee_enrol()
    elif file == 'attain':
        data = __lee_attain()
    else:
        raise ValueError(
            "Invalid code parameter. Please pass one of the parameters 'hc', 'enrol', 'attain'.")

    if date is not None:
        data = data[data['date'] >= date]

    data['Country'] = data['Country'].str.upper()

    data['date'] = pd.to_datetime(data.date.astype(str))

    data = data.pivot(index='date', columns='Country', values=code)

    return data


def __date_control_quarter(x):
    dates = x.dropna().index.to_series()

    start = dates.diff() < offset.Day(95)
    end = dates.shift(-1) - dates < offset.Day(95)

    start = dates[~start].values
    end = dates[~end].values

    start = pd.PeriodIndex(start, freq='Q')
    end = pd.PeriodIndex(end, freq='Q')

    return list(zip(start, end))


def __date_control(x):
    dates = x.dropna().index.to_series()

    start = dates.diff() != 1

    end = dates.shift(-1) - dates != 1

    start = dates[start].values
    end = dates[end].values

    return list(zip(start, end))


def control(data, freq: str = 'a', name=None):
    """
    Verileri kontrol etmek için kullanılır.

    Args:
        data (Dataframe or Serie) : Kontrol edilmek istenen veri.
        freq (str) : Default 'a'. Eğer quarterly veriler kontrol edilmek isteniyorsa 'q' ayarlanmalı.
        name (str) : Kontrol sonucunu adlandırmak için kullanılır.
    """
    df = data.copy()

    if freq == 'q':
        df_year = df.apply(__date_control_quarter)
    else:
        df.index = df.index.to_series().dt.year
        df_year = df.apply(__date_control)
    if isinstance(df_year, pd.DataFrame):
        df_year = df_year.loc[0]
    df_result = pd.concat([df_year, df.count()], axis=1)
    df_result.columns = ['start-end', 'total']
    if name is not None:
        df_result['name'] = name
    df_result.to_csv('data/app/control.csv')
    return df_result
