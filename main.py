from itertools import count

import lmts
import data
import pandas as pd

# source değişkenleri
source_imf = data.source('imf')
source_bl = data.source('bl')
#source_wb = data.source('wb')
source_pwt = data.source('pwt')
source_eora = data.source('eora')
source_oecd = data.source('oecd')
#source_woid = data.source('woid')
#source_tivan = data.source('tivan')

# reading data from sources (syntax sre somewhat different in oecd)
# and control their availability for different countries
# pop = data.read_wb('SP.POP.GROW', date=1950)
# data_control = data.control(pop)
# gdp_imf = data.read_imf('NGDP_XDC', 'q', date=1970)
# data_control_1 = data.control(gdp_imf,freq='q')

#gdp_oecd = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')
#data_control_2 = data.control(gdp_oecd,freq='q')

# PWT DATA, YEARLY
# human capital index values
hc = data.read_pwt('hc', date=1950)
# population in millions
pop = data.read_pwt('pop', date=1950)
# capital stock constant prices
cap = data.read_pwt('rnna', date=1950)
# capital stock constant prices
gdp = data.read_pwt('rgdpna', date=1950)
cap_share = 10*(cap/gdp)
p = data.read_pwt('pl_c', date=1950)
# Barro-Lee HC Data, 5 years interval
# hc_s = data.read_bl(code='attain_No Schooling', date=1950)

gvc = data.read_eora('gvc', date=1950)
gexp = data.read_eora('gexp', date=1950)
gvcp_eora = gvc / gexp


# X variable vector
X = [gvcp_eora, hc, pop, cap]

# liste halindeki X'i data frame aktar
names = ['gvcp_eora', 'hc', 'pop', 'cap']

merged_X = pd.concat(X, keys=names)

# reshape X
merged_X = merged_X.stack().unstack(0)

# 0 olanlari nan yap
merged_X[merged_X.eq(0)] = None

# Creating growth variables and add to the data frame
#growth = lmts.growth(merged_X[['gdp',' ', 'p']])
#merged_X[['g_gdp','g_pop', 'g_p' ]] = growth.stack()

# eksizsiz panel olusturmak icin nan lari at
merged_X = merged_X.dropna()

# time average al
average = merged_X[['gvcp_eora', 'hc', 'pop', 'cap']].groupby(level=1).mean()
# average.to_csv('average.csv')

#d_data = data.read_pwt('rgdpna', date=1950)
d_data = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')

# Ln hesaplama
d_data = lmts.ln(d_data)

# data-USA
d_data = lmts.diff(d_data, 'mean', drop=False)
# d_data = lmts.diff(d_data, 'USA', drop=True)
# 48'den az veri içeren ülkeleri kaldır
# d_data = lmts.constrain(d_data, 48)

# d'leri hesapla
d_values = lmts.get_d_values(d_data)

average[['elw_m', 'elw_n', 'elw2s_v0', 'elw2s_v1', 'elw2s_h0', 'elw2s_h1', 'gph',
       'hou_perron', 'local_w']] = d_values

# eksizsiz panel olusturmak icin butun tarihker icin ortak ulkeleri bul
# ci = lmts.country_intersection(merged_X)
# average = average[average.index.isin(ci)]

average = average.dropna()

# write csv
average.reset_index()\
    .assign(date=merged_X.reset_index().date.dt.year)\
    .to_csv('x_values_balanced.csv', index=False)



# gdp_90 = data.read_pwt('rgdpna', date=1990)

# d_data = gdp_90.copy()
# reading data for computing d's
# d_data = data.read_imf('NGDP_R_SA_XDC', 'q', date=1990)
# d_data = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')

# y = d_values.local_w
#
# average['d'] = y
#
# average = average.dropna()
#
# average.to_csv('avg.csv')

# WB Data, YEARLY, all variables are (%)
# population growth rate
pop = data.read_wb('SP.POP.GROW', date=1950)
# inflation rate
inf = data.read_wb('FP.CPI.TOTL.ZG', date=1950)
# gross capital formation % of GDP
inv = data.read_wb('NE.GDI.TOTL.ZS', date=1950)
# government debt % of GDP
govdebt = data.read_wb('GC.DOD.TOTL.GD.ZS', date=1950)


# Creating global value chain variables

gvc = data.read_woid('gvc', date=1950)
gexp = data.read_woid('gexp', date=1950)
gvcp_woid = gvc / gexp

gvc = data.read_eora('gvc', date=1950)
gexp = data.read_eora('gexp', date=1950)
gvcp_eora = gvc / gexp

gvc = data.read_tivan('gvc', date=1950)
gexp = data.read_tivan('gexp', date=1950)
gvcp_tivan = gvc / gexp


######


# X variable vector
X = ['gvcp_eora', 'hc', 'pop', 'cap']

# take logarithms
# gvcp_eora = lmts.ln([gvcp_eora])

# Birden fazla değişken için USA farkı hesaplama
# gvcp_eora = lmts.diff([gvcp_eora], 'USA', drop=True)

gvcp_eora, hc, pop, cap = lmts.mean([gvcp_eora, hc, pop, cap])

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