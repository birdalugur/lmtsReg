import lmts
import data

# source değişkenleri
source_imf = data.source('imf')
source_bl = data.source('bl')
source_wb = data.source('wb')
source_pwt = data.source('pwt')
source_eora = data.source('eora')
source_oecd = data.source('oecd')

# d'leri hesaplamak için veri okunuyor
d_data = data.read_imf('NGDP_XDC', 'q', date=1970)

# Ln hesaplama
d_data = lmts.ln(d_data)

# data-USA
d_data = lmts.diff(d_data, 'USA', drop=True)

# 48'den az veri içeren ülkeleri kaldır
d_data = lmts.constrain(d_data, 48)

# d'leri hesapla
d_values = lmts.get_d_values(d_data)

# Read X values
dataoecd = data.oecd(frequency='q', measure='IDX', subject='VOLIDX')
datawoid = data.read_woid('daa', date=1950)
eora_gvc = data.read_eora('gvc', date=1950)
eora_gexp = data.read_eora('gexp', date=1950)
eora = eora_gvc / eora_gexp
imfdata = data.read_imf('BFXF_BP6_USD', 'a', date=1950)
wbdata = data.read_wb('SP.POP.GROW', date=1950)
bldata = data.read_bl(code='attain_No Schooling', date=1950)
pwtdata = data.read_pwt(code='statcap', date=None)

# Birden fazla değişken için ln alma
imfdata, pwtdata = lmts.ln([imfdata, pwtdata])

# Birden fazla değişken için USA farkı hesaplama
wbdata, bldata = lmts.diff([wbdata, bldata], 'USA', drop=True)

# Regresyon için independent değişkenleri seçme
X = [eora, pwtdata, wbdata]

# Ortalamaları alma
X = lmts.mean(X)

# y_true
y_true = d_values.elw_m

# kesişim
X, y = lmts.intersection(X, y_true)

# tahmin için test verisi oluşturma
X_test = lmts.test_data(X)

# Regresyon
model = lmts.Model(X, y, X_test)
model.plot()
