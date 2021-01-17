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

# gdp_oecd = data.oecd(frequency='q', measure='IDX', subject='VOLIDX', date=1955)
# data_control_2 = data.control(gdp_oecd,freq='q')

# reading data for computing d's
# d_data = data.read_imf('NGDP_XDC', 'q', date=1970)
# d_data = data.read_pwt('rgdpna', date=1970)
d_data = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')

# Ln hesaplama
d_data = lmts.ln(d_data)

# data-USA
d_data = lmts.diff(d_data, 'mean', drop=False)

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

tivan_gvc = data.read_tivan('gvc', date=1950)
tivan_gexp = data.read_tivan('gexp', date=1950)
tivan = tivan_gvc / tivan_gexp

hc = data.read_bl('hc_Human Capital', date=1950)
# hc_s = data.read_bl(code='attain_No Schooling', date=1950)
pop = data.read_wb('SP.POP.GROW', date=1950)
inf = data.read_wb('FP.CPI.TOTL.ZG', date=1950)
cap = data.read_wb('NE.GDI.TOTL.ZS', date=1950)
govdebt = data.read_wb('GC.DOD.TOTL.GD.ZS', date=1950)

# take logarithms
# imfdata, pwtdata = lmts.ln([imfdata, pwtdata])

# Birden fazla değişken için USA farkı hesaplama
# gvcp_eora, hc, pop, inf, cap, govdebt = lmts.diff([gvcp_eora, hc, pop, inf, cap, govdebt], 'USA', drop=True)

# Regresyon için independent değişkenleri seçme
X = [gvcp_eora, hc, pop, inf, cap, govdebt]

# X değişkenlerini csv'ye yazma
names = ['gvcp_eora', 'hc', 'pop', 'inf', 'cap', 'govdebt']
merged_X = pd.concat(X, keys=names)

# write csv
merged_X.to_csv('X_values')


def initial_func(x): return x.dropna().groupby(level=0).resample('Y', level='date').apply(lambda x: x.head(1))


initial = merged_X.apply(initial_func)

# Ortalamaları alma
X = lmts.mean(X)

# y values
y = d_values.local_w

# intersection of countries
X, y = lmts.intersection(X, y)

# Regression

import statsmodels.api as sm

ols = sm.OLS(y, X)
ols_result = ols.fit()
std_error = ols_result.bse
ols_result.summary()

# Scatter Plot with several X
# Fitted vaues at means of X

X_test = lmts.test_data(X)
model = lmts.Model(X, y, X_test)
model.plot()
