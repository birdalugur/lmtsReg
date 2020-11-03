#!/usr/bin/env python
# coding: utf-8


import os
import inspect

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go

import data

os.mkdir('data/app')

def retrieve_name(var):
    """
    Getting the name of a variable as a string"""

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def ln_diff(data, country='USA'):
    """
    ln_diff = ln(X) - ln(USA) and remove 'USA' column."""

    ln_data = np.log(data)
    ln_data = ln_data.sub(ln_data[country], axis=0)
    return ln_data.drop(country, axis=1)


def constrain(data, n):
    """
    n'den veri içeren sütunları kaldır."""

    data = data[data.columns[data.count() >= n]]
    return data


def get_d_values(file):
    """
    Run the R script and get the results."""
    os.system('Rscript dvals.R {}'.format(file))
    return pd.read_csv('data/app/d_values.csv', index_col=0)


# d'leri hesapla
df = data.pwt()
df = ln_diff(df)
df = constrain(df, 48)
df.to_csv('data/app/lndiff.csv')
d_values = get_d_values('data/app/lndiff.csv')


# data.indicator('imf')
# data.indicator('wb')
# data.indicator('bl')


# Read X values
eora = data.from_eora(date=1950)
woid = data.from_WOID(date=1950)

imfdata = data.read_imf('BFXF_BP6_USD', 'a', date=1950)
wbdata = data.read_wb('SP.POP.GROW', date=1950)

bldata = data.read_BL(code='attain', variable='No Schooling', date=1950)


# ln(x)-ln(usa) farklarının hesaplanmasını istediğimiz değişkenleri burada belirtiyloruz
ln_list = [bldata, wbdata]


for var in ln_list:
    print(retrieve_name(var))
    globals()[retrieve_name(var)] = ln_diff(var)


# Select X and y
X = [eora, woid, wbdata]

y = d_values.elw_m


X = list(map(lambda x: x.mean(), X))
X = pd.concat(X, axis=1).dropna()
countries = y.index.intersection(X.index)

X = X.loc[countries]
y = y[countries]


# Test Verisi
test = X.copy()
average = X.mean()[1:]
for i in average.index:
    test[i] = average[i]

test = test.sort_values(0)


# Lineer Regresyon kullanarak tahmin
regr = LinearRegression()
regr.fit(X, y)
test['y_pred'] = regr.predict(test)


coefficients = regr.coef_
intercept = regr.intercept_
r_square = r2_score(y.reindex(test.index), test['y_pred'])
print('coefficients:\n', coefficients)
print('intercept:\n', intercept)
print('r square:\n', r_square)


# PLOT
fig = px.scatter(x=X[0], y=y, text=countries)
fig.add_traces(go.Scatter(x=test[0], y=test['y_pred'], name='Regression Fit'))
offline.plot(fig, filename='data/app/regression.html')
