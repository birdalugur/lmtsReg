#!/usr/bin/env python
# coding: utf-8


import os
import subprocess
import inspect

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go


def create_appdata():
    try:
        os.mkdir('data/app')
    except FileExistsError:
        pass


def retrieve_name(var):
    """
    Getting the name of a variable as a string"""

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def ln(df):
    """
    Doğal logaritmayı hesaplar."""
    if isinstance(df, list):
        ln_list = []
        for _df in df:
            ln_list.append(np.log(_df))
        return ln_list
    return np.log(df)


def diff(df, country, drop=False):
    """
    Example: X-USA"""

    if isinstance(df, list):
        diff_list = []
        for _df in df:
            if country == 'mean':
                diff_value = _df.mean(axis=1)
            else:
                diff_value = _df[country]
            diff = _df.sub(diff_value, axis=0)
            if drop:
                diff = diff.drop(country, axis=1)
            diff_list.append(diff)
        return diff_list
    if country == 'mean':
        diff_value = df.mean(axis=1)
    else:
        diff_value = df[country]
    diff = df.sub(diff_value, axis=0)
    if drop:
        diff = diff.drop(country, axis=1)
    return diff


def constrain(data, n):
    """
    n'den veri içeren sütunları kaldır."""

    data = data[data.columns[data.count() >= n]]
    return data


def get_d_values(df):
    """
    Run the R script and get the results."""
    file = 'data/app/lndiff.csv'
    df.to_csv('data/app/lndiff.csv')

    if os.name == 'posix':
        command = 'Rscript dvals.R {}'.format(file)
        os.system(command)
    else:
        command = 'C:/Program Files/R/R-3.6.3/bin/x64/Rscript dvals.R {}'.format(file)
        subprocess.call(command)

    try:
        d = pd.read_csv('data/app/d_values.csv', index_col=0)
    except FileNotFoundError:
        raise Exception("Please update your R path.")

    return d


def mean(data):
    if isinstance(data, list):
        return list(map(lambda x: x.mean(), data))
    return data.mean()


def intersection(X, y):
    X = pd.concat(X, axis=1).dropna()
    countries = y.index.intersection(X.index)
    X = X.loc[countries]
    y = y[countries]
    return X, y


def test_data(X):
    # Test Verisi
    test = X.copy()
    average = X.mean()[1:]
    for i in average.index:
        test[i] = average[i]

    test = test.sort_values(0)
    return test


class Model:
    __model = LinearRegression()

    def __init__(self, X_true, y_true, X_test):
        self.training_data = X_true
        self.target_values = y_true
        self.test_values = X_test
        self.__fit()

    def __fit(self):
        self.__model.fit(self.training_data, self.target_values)

    def predict(self):
        return self.__model.predict(self.test_values)

    @property
    def intercept(self):
        return self.__model.intercept_

    @property
    def r_square(self):
        # TODO: r2 hesaplanacak
        pass

    @property
    def cofficients(self):
        return self.__model.coef_

    def plot(self) -> go.Figure:
        fig = px.scatter(x=self.training_data[0], y=self.target_values)
        fig.add_traces(go.Scatter(x=self.test_values[0], y=self.predict(), name='Regression Fit'))
        offline.plot(fig, filename='data/app/regression.html')


if __name__ == 'lmts':
    # data/app klasörünü oluşturur.
    create_appdata()
