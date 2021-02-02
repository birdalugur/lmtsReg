#!/usr/bin/env python
# coding: utf-8


import os
import subprocess
import inspect
from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go


def create_appdata():
    """
    Data dizini altında app klasörü oluşturur.
    app klasöründe uygulama verileri saklanır.
    """
    try:
        os.mkdir('data/app')
    except FileExistsError:
        pass


def retrieve_name(var):
    """
    Getting the name of a variable as a string"""

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def ln(data: Union[List[pd.DataFrame], pd.DataFrame]) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Doğal logaritmayı hesaplar.

    Args:
        data (Dataframe or list)
    Returns:
        Dataframe or List
    """
    if isinstance(data, list):
        ln_list = []
        for _df in data:
            ln_list.append(np.log(_df))
        return ln_list
    return np.log(data)


def diff(df: pd.DataFrame, subtrahend: str, drop: bool = False) -> Union[pd.DataFrame, list]:
    """
    Verinin subtrahend ile belirtilen sütundan farkını alın.

    Args:
        df : Dataframe or List
        subtrahend : Çıkarılacak sütun adı. Ortalama için 'mean' ayarlanmalı.
        drop: subtrahend ile belirtilen sütun dataframe'den çıkarılsın mı?

    Returns:
        Union[pd.DataFrame, list]
    """

    if isinstance(df, list):
        diff_list = []
        for _df in df:
            if subtrahend == 'mean':
                diff_value = _df.mean(axis=1)
            else:
                diff_value = _df[subtrahend]
            diff = _df.sub(diff_value, axis=0)
            if drop:
                diff = diff.drop(subtrahend, axis=1)
            diff_list.append(diff)
        return diff_list
    if subtrahend == 'mean':
        diff_value = df.mean(axis=1)
    else:
        diff_value = df[subtrahend]
    diff = df.sub(diff_value, axis=0)
    if drop:
        diff = diff.drop(subtrahend, axis=1)
    return diff


def constrain(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Belirtilen sayıdan az veri içeren sütunları DataFrame'den kaldırır.

    Args:
        data: Kısıta göre yeniden düzenlenecek veri.
        n: kısıt sayısı

    Returns:
        pd.DataFrame
    """

    data = data[data.columns[data.count() >= n]]
    return data


def get_d_values(df: pd.DataFrame):
    """
    Run the R script and get the results.
    R'ın LongMemoryTS paketini kullanarak d'leri hesaplar.
    Rscript çalıştırır ve sonuçları döndürür.

    Args:
        df : Date x Country

    Returns: 'elw_m', 'elw_n', 'elw2s_v0', 'elw2s_v1', 'elw2s_h0', 'elw2s_h1', 'gph',
       'hou_perron', 'local_w' verileri içeren Dataframe

    """
    file = 'data/app/lndiff.csv'
    df.to_csv('data/app/lndiff.csv')

    if os.name == 'posix':
        command = 'Rscript dvals.R {}'.format(file)
        os.system(command)
    else:
        command = 'C:/Program Files/R/R-4.0.3/bin/x64/Rscript dvals.R {}'.format(file)
        subprocess.call(command)

    try:
        d = pd.read_csv('data/app/d_values.csv', index_col=0)
    except FileNotFoundError:
        raise Exception("Please update your R path.")

    return d


def mean(data: Union[List[pd.DataFrame], pd.DataFrame]):
    """
    Aritmetik ortalamayı hesaplar.

    Returns: DataFrame or List of DataFrame
    """
    if isinstance(data, list):
        return list(map(lambda x: x.mean(), data))
    return data.mean()


def intersection(x: list, y:pd.Series):
    """
    Ülkelerin kesişimini alın.

    Args:
        x: (list of series) : Regresyonda kullanılacak x değerlerinin listesi.
        y: pd.Series

    Returns: new x and y
    """
    x = pd.concat(x, axis=1).dropna()
    countries = y.index.intersection(x.index)
    x = x.loc[countries]
    y = y[countries]
    return x, y



def country_intersection(merged_X):
    interlist = list(merged_X.groupby(level='date'))

    x = interlist[0][1]
    x = x.index.get_level_values(1)

    for i in range(1, len(interlist)):
        y = interlist[i][1]
        y = y.index.get_level_values(1)
        x = x.intersection(y)

    return x



def initial_values(data):

    x = data.unstack(0)

    x.iloc[:, 0]

    for col in x.columns:
        x[col] = x.iloc[:, 0]
    return  x


def buyume_orani(data):
    x = data.unstack(1)
    return x.diff() / x


def test_data(x):
    """
    Eğitim verilerinden kullanarak test verisi oluştur.

    Args:
        x: (pd.DataFrame) : Training data

    Returns:
        Test Data
    """
    test = x.copy()
    average = x.mean()[1:]
    for i in average.index:
        test[i] = average[i]

    test = test.sort_values(0)
    return test


class Model:
    """
    En küçük kareler yöntemini kullanan Doğrusal Regresyon işlemlerini gerçekleştirin.

    Args:
        training_data: (Dataframe): Kullanılacak X'leri içermelidir.
        target_values: (Series): Kullanılacak d_values.
        test_values: test verileri.

    """
    __model = LinearRegression()

    def __init__(self, training_data, target_values, test_values):
        self.training_data = training_data
        self.target_values = target_values
        self.test_values = test_values
        self.__fit()

    def __fit(self):
        self.__model.fit(self.training_data, self.target_values)

    def predict(self):
        return self.__model.predict(self.test_values)

    @property
    def intercept(self):
        """
        y ekseni kesim noktası
        """
        return self.__model.intercept_

    @property
    def countries(self):
        """
        Regresyonda kullanılan ülkeler
        """
        return self.training_data.index.values

    @property
    def r_square(self):
        # TODO: r2 hesaplanacak
        pass

    @property
    def cofficients(self):
        """
        tahmini katsayılar
        """
        return self.__model.coef_

    def plot(self) -> go.Figure:
        """
        Regresyon grafiğini çizer ve data/app dizininde regression.html olarak kaydeder.
        """
        fig = px.scatter(x=self.training_data[0], y=self.target_values, text=self.countries)
        fig.add_traces(go.Scatter(x=self.test_values[0], y=self.predict(), name='Regression Fit'))
        offline.plot(fig, filename='data/app/regression.html')


if __name__ == 'lmts':
    # data/app klasörünü oluşturur.
    create_appdata()
