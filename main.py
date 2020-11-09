import lmts
import data

# indicator değişkenleri
indicator_imf = data.indicator('imf')
indicator_bl = data.indicator('bl')
indicator_wb = data.indicator('wb')

# d'leri hesaplamak için veri okunuyor
d_data = data.read_imf('NGDP_XDC', 'q', date=1970)

# Ln hesaplama
d_data = lmts.ln(d_data)

# data-USA
d_data = lmts.diff(d_data, 'USA')

# 48'den az veri içeren ülkeleri kaldır
d_data = lmts.constrain(d_data, 48)

# d'leri hesapla
d_values = lmts.get_d_values(d_data)

# Read X values
eora = data.from_eora(date=1950)
woid = data.from_woid(date=1950)
imfdata = data.read_imf('BFXF_BP6_USD', 'a', date=1950)
wbdata = data.read_wb('SP.POP.GROW', date=1950)
bldata = data.read_bl(code='attain', variable='No Schooling', date=1950)

# Birden fazla değişken için ln alma
imfdata, wbdata, woid = lmts.ln([imfdata, wbdata, woid])

# Birden fazla değişken için USA farkı hesaplama
wbdata, woid = lmts.diff([wbdata, woid], 'USA')


# Regresyon için independent değişkenleri seçme
X = [eora, woid, wbdata]

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

