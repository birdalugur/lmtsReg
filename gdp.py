from itertools import count

import lmts
import data
import pandas as pd

# source değişkenleri
source_imf = data.source('imf')
source_bl = data.source('bl')
source_wb = data.source('wb')
source_pwt = data.source('pwt')
source_eora = data.source('eora')
source_oecd = data.source('oecd')
source_woid = data.source('woid')
source_tivan = data.source('tivan')

# reading data from sources (syntax sre somewhat different in oecd)
# and control their availability for different countries
# pop = data.read_wb('SP.POP.GROW', date=1950)
# data_control = data.control(pop)
# gdp_imf = data.read_imf('NGDP_XDC', 'q', date=1970)
# data_control_1 = data.control(gdp_imf,freq='q')

#gdp_oecd = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')
#data_control_2 = data.control(gdp_oecd,freq='q')

# reading data for computing d's
# d_data = data.read_imf('NGDP_XDC', 'q', date=1970)
d_data = data.read_pwt('rgdpna', date=1970)
#d_data = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')

# Ln hesaplama
d_data = lmts.ln(d_data)

# data-USA
d_data = lmts.diff(d_data, 'mean', drop=True)

# 48'den az veri içeren ülkeleri kaldır
d_data = lmts.constrain(d_data, 48)

# d'leri hesapla
d_values = lmts.get_d_values(d_data)

# Read X values

gvc = data.read_woid('gvc', date=1950)
gexp = data.read_woid('gexp', date=1950)
gvcp_woid = gvc / gexp

gvc = data.read_eora('gvc', date=1950)
gexp = data.read_eora('gexp', date=1950)
gvcp_eora = gvc / gexp

gvc = data.read_tivan('gvc', date=1950)
gexp = data.read_tivan('gexp', date=1950)
gvcp_tivan = gvc / gexp

hc = data.read_pwt('hc', date=1950)
# data_control = data.control(inf)
# hc_s = data.read_bl(code='attain_No Schooling', date=1950)
pop = data.read_wb('SP.POP.GROW', date=1950)
inf = data.read_wb('FP.CPI.TOTL.ZG', date=1950)
cap = data.read_wb('NE.GDI.TOTL.ZS', date=1950)
govdebt = data.read_wb('GC.DOD.TOTL.GD.ZS', date=1950)
gdp = data.read_pwt('rgdpna', date=1950)

# take logarithms
# imfdata, pwtdata = lmts.ln([imfdata, pwtdata])

# Birden fazla değişken için USA farkı hesaplama
# gvcp_eora, hc, pop, inf, cap, govdebt = lmts.diff([gvcp_eora, hc, pop, inf, cap, govdebt], 'USA', drop=True)

# Regresyon için independent değişkenleri seçme
X = [gvcp_eora, gdp, pop, cap]

names = ['gvcp_eora', 'gdp', 'pop', 'cap']

merged_X = pd.concat(X, keys=names)

merged_X = merged_X.stack().unstack(0)

merged_X[merged_X.eq(0)] = None

merged_X = merged_X.dropna()

ci = lmts.country_intersection(merged_X)


merged_X = merged_X[merged_X.index.get_level_values(1).isin(ci)]

# write csv
merged_X.reset_index()\
    .assign(date=merged_X.reset_index().date.dt.year)\
    .to_csv('x_values.csv',index=False)


init_gdp = lmts.initial_values(merged_X)


average = merged_X[['pop','gdp']]
average = average.groupby(level=1).mean()
average.to_csv('average.csv')

init_gdp

lmts.buyume_orani(merged_X[['gdp', 'pop']])
# Ortalamaları alma



# Ortalamaları alma

X = lmts.mean(X)

# y values
y = d_values.local_w

# intersection of countries
X, y = lmts.intersection(X, y)

# Regression

import statsmodels.api as sm
endog = y
exog = sm.add_constant(X)

# Fit and summarize OLS model
ols = sm.OLS(endog, exog)
ols_result = ols.fit()
std_error =ols_result.bse
ols_result.summary()

# Scatter Plot with several X
# Fitted vaues at means of X

X_test = lmts.test_data(X)
model = lmts.Model(X, y, X_test)
model.plot()

