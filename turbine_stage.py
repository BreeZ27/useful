import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
pi=math.pi


def tcalc(p0=8.28 * 10 ** 5, pk=6 * 10 ** 5, T0=293, d=0.195, l=0.015, e=6 / 28, Qin=None, n=50, use_xf=False,
          effgen=0.8):
    ro = 0.1
    fi = 0.97
    mu = 0.97
    ksi = 0.97
    xvs = 0

    d = d
    l = l
    a1 = 12
    e = e
    n = n
    effm = 0.98
    effgen = effgen

    k = 1.3
    R = 512
    rny = 0.7
    m = (k - 1) / k
    Cp = R * k / (k - 1)

    p0 = p0
    pk = pk
    T0 = T0

    xf = fi * math.cos(a1 * pi / 180) / (2 * (1 - ro) ** (0.5))

    delt = p0 / pk
    H0 = Cp * T0 * (1 - delt ** (-m))
    h0 = Cp * T0
    ro0 = p0 / (R * T0)
    H0s = H0 * (1 - ro)
    H0r = H0 * ro

    c1t = (2 * H0s) ** (0.5)
    h1t = h0 - H0s
    T1t = h1t / Cp
    a = (k * R * T1t) ** (0.5)
    M = c1t / a

    p1 = p0 * (T1t / T0) ** (1 / m)
    ro1t = p1 / (R * T1t)
    F1 = pi * d * l * e * math.sin(a1 * pi / 180)
    G = F1 * ro1t * mu * c1t
    Q_ = G / rny * 3600

    if Qin != None:

        while Q_ > Qin:
            p0 = p0 - 0.05 * 10 ** 5
            delt = p0 / pk
            H0 = Cp * T0 * (1 - delt ** (-m))
            h0 = Cp * T0
            ro0 = p0 / (R * T0)
            H0s = H0 * (1 - ro)
            H0r = H0 * ro

            c1t = (2 * H0s) ** (0.5)
            h1t = h0 - H0s
            T1t = h1t / Cp
            a = (k * R * T1t) ** (0.5)
            M = c1t / a

            p1 = p0 * (T1t / T0) ** (1 / m)
            ro1t = p1 / (R * T1t)
            F1 = pi * d * l * e * math.sin(a1 * pi / 180)
            G = F1 * ro1t * mu * c1t
            Q_ = G / rny * 3600

    if use_xf == True:
        n = round(xf * ((2 * H0) ** (0.5)) / pi / d, 0)

    c1 = c1t * fi
    Hs = c1 * c1 / 2
    DHs = (c1t * c1t - c1 * c1) / 2
    h1 = h0 - Hs
    T1 = h1 / Cp
    ro1 = p1 / (R * T1)

    h2t = h1 - H0r
    T2t = h2t / Cp
    ro2t = pk / (R * T2t)

    T2t_ = T0 * (delt) ** (-m)
    h2t_ = Cp * T2t_
    r2rt_ = pk / (R * T2t_)

    u = pi * d * n
    w1 = (c1 * c1 + u * u - 2 * c1 * u * math.cos(a1 * pi / 180)) ** (0.5)
    bet1 = math.atan((c1 * math.sin(a1 * pi / 180)) / (c1 * math.cos(a1 * pi / 180) - u)) * 180 / pi
    w2t = (w1 * w1 + 2 * H0r) ** (0.5)
    w2 = w2t * ksi
    DHr = w2t * w2t / 2 * (1 - ksi * ksi)

    h2 = h2t + DHr
    T2 = h2 / Cp
    ro2 = pk / (R * T2)

    F2 = G / (ro2t * mu * w2t)
    bet2 = math.asin(F2 / (e * pi * d * l)) * 180 / pi
    a2 = math.atan(w2 * math.sin(bet2 * pi / 180) / (w2 * math.cos(bet2 * pi / 180) - u)) * 180 / pi

    if a2 < 0:
        a2 = 90 + (90 - abs(a2))

    c2 = (w2 * w2 + u * u - 2 * w2 * u * math.cos(bet2 * pi / 180)) ** (0.5)
    DHvs = c2 * c2 / 2

    Lu = (c1 * c1 - c2 * c2) / 2 + (w2 * w2 - w1 * w1) / 2
    Lu1 = u * (w1 * math.cos(bet1 * pi / 180) + w2 * math.cos(bet2 * pi / 180))
    Lu2 = H0 - DHs - DHr - DHvs

    if (round(Lu - Lu1, 3) != 0) or (round(Lu - Lu2, 3) != 0):
        print('Check Lu key')

    Ru = G * (c1 * math.cos(a1 * pi / 180) + c2 * math.cos(a2 * pi / 180))
    Ru1 = G * (w1 * math.cos(bet1 * pi / 180) + w2 * math.cos(bet2 * pi / 180))

    if round(Ru - Ru1, 3) != 0:
        print('Check Ru key')

    E0 = H0 - xvs * DHvs
    hol = Lu / E0
    Nu = Lu * G
    Mkr = Nu / (2 * pi * n)

    DHparc = 0.016 * H0
    DHy = 0.04 * H0

    Hi = E0 - DHr - DHs - (1 - xvs) * DHvs - DHy - DHparc
    Ne = Hi * G * effm * effgen
    ny = Ne / (H0 * G)

    flow = pd.DataFrame(np.array([[p0, T0, h0], [p1, T1, h1], [pk, T2, h2]]), columns=['Pressure', 'Temp', 'Enthalpy'])
    PwrOut = pd.DataFrame(np.array([[Ne, Mkr, Q_, ny]]), columns=['Power', 'Torque', 'VolumeFlow', 'Eff'])

    return flow, PwrOut

df = pd.read_csv('massflow_data.csv', sep=';')
df.columns = ['date', 't0', 'p0', 'tkp', 'pk', 'Q']
df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
df['month'] = df['date'].apply(lambda x: x.strftime('%Y-%m'))

df['pk'] = df['pk'].apply(lambda x: x.replace(',', '.'))
df['p0'] = df['p0'].apply(lambda x: x.replace(',', '.'))
df['pk'] = df['pk'].astype('float64')
df['p0'] = df['p0'].astype('float64')

df['tkp'] = df['tkp'].apply(lambda x: x.replace(',', '.')).astype('float64')
df['t0'] = df['t0'].astype('float64')
# df.head()


df['Power'] = df[df['Q'] > 550]['Q'].apply(lambda x: tcalc(Qin=x)[1]['Power'])
df.loc[df['Power'] < 1500, 'Power'] = None
# df

# fig, axes = plt.subplots(nrows=len(df['month'].unique()), ncols=1, figsize=(15, 120))
# i = 0
# for month, data in df.groupby('month'):
#     data = data.sort_values(by='date')
#     k = 0
#
#     line1 = axes[i]
#     line1.plot(data['date'], data['Q'], alpha=0.5, label='Расход')
#
#     line2 = axes[i].twinx()
#     line2.plot(data['date'], data['Power'], color='g', linestyle='--', label='Мощность ТДУ')
#     axes[i].set_ylabel('Объёмный расход [м^3/ч]', fontsize=12)
#     axes[i].set_xlabel('Дата', fontsize=12)
#     axes[i].grid()
#     line2.axhline(5000, color='y', linestyle='--', label='Уровень мощности 5 кВт')
#     line2.set_ylabel('Мощность [Вт]', fontsize=12)
#     plt.legend()
#     line2.set_yticks(list(range(0, 7000, 1000)))
#
#     i += 1
#
# plt.tight_layout()
# plt.savefig('Объёмный расход.jpg')
# plt.show()


df['Power_nt'] = df[df['Q'] > 500]['Q'].apply(lambda x: tcalc(p0=7.55*10**5, T0=293, d=0.118, l=0.012, e=6/28, \
                                                              n=50, use_xf=True, Qin=x, effgen=0.9)[1]['Power'])
df.loc[df['Power_nt'] < 1000, 'Power_nt'] = None
# df

# fig, axes = plt.subplots(nrows=len(df['month'].unique()), ncols=1, figsize=(15, 120))
# i = 0
# for month, data in df.groupby('month'):
#     data = data.sort_values(by='date')
#     k = 0
#     line1 = axes[i]
#     line1.plot(data['date'], data['Q'], alpha=0.5, label='Расход')
#
#     line2 = axes[i].twinx()
#     line2.plot(data['date'], data['Power_nt'], color='g', linestyle='--', label='Мощность ТДУ')
#     axes[i].set_ylabel('Объёмный расход [м^3/ч]', fontsize=12)
#     axes[i].set_xlabel('Дата', fontsize=12)
#     axes[i].grid()
#     line2.axhline(5000, color='y', linestyle='--', label='Уровень мощности 5 кВт')
#     line2.set_ylabel('Мощность [Вт]', fontsize=12)
#     plt.legend()
#     line2.set_yticks(list(range(0, 7000, 1000)))
#
#     i += 1
# plt.tight_layout()
# plt.savefig('Объёмный расход_2.jpg')
# plt.show()


# Выработка кВт*ч
W_h = df['Power'].apply(lambda x: x/1000*2).sum().round(0)
W_h_nt = df['Power_nt'].apply(lambda x: x/1000*2).sum().round(0)
print(W_h)
print(W_h_nt)
print(round(W_h_nt/W_h, 3))


# Время работы с номальной мощностью для ТДА с общепром ген.
FPwt = df[df['Power'] > 5000].count()['p0']*2
# Время работы с номальной мощностью для ТДА с вент-реакт ген.
FPwt_nt = df[df['Power_nt'] > 5000].count()['p0']*2
print(FPwt)
print(FPwt_nt)
print(round(FPwt_nt/FPwt, 3))

for month, data in df.groupby('month'):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    fig.suptitle('Зависимость выработаки электроэнергии от расхода в ' + month, fontsize=16)

    data = data.sort_values(by='date')
    i = 0

    line1 = axes[i]
    line1.plot(data['date'], data['Q'], alpha=0.5, label='Расход')

    line2 = axes[i].twinx()
    line2.plot(data['date'], data['Power'], color='g', linestyle='--', label='Мощность ТДУ М1')
    axes[i].set_ylabel('Объёмный расход [м^3/ч]', fontsize=12)
    axes[i].set_xlabel('Дата', fontsize=12)
    axes[i].grid()
    line2.axhline(5000, color='y', linestyle='--', label='Уровень мощности 5 кВт')
    line2.set_ylabel('Мощность [Вт]', fontsize=12)
    plt.legend()
    line2.set_yticks(list(range(0, 7000, 1000)))

    i = 1

    line1 = axes[i]
    line1.plot(data['date'], data['Q'], alpha=0.5, label='Расход')

    line2 = axes[i].twinx()
    line2.plot(data['date'], data['Power_nt'], color='g', linestyle='--', label='Мощность ТДУ М2')
    axes[i].set_ylabel('Объёмный расход [м^3/ч]', fontsize=12)
    axes[i].set_xlabel('Дата', fontsize=12)
    axes[i].grid()
    line2.axhline(5000, color='y', linestyle='--', label='Уровень мощности 5 кВт')
    line2.set_ylabel('Мощность [Вт]', fontsize=12)
    plt.legend()
    line2.set_yticks(list(range(0, 7000, 1000)))

    plt.savefig(month + ' Объёмный расход_3.jpg')
    plt.show()
