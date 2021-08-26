import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time, os
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor as PPE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE

import warnings
warnings.filterwarnings('ignore')

plt.rc('font', family='Malgun Gothic')
plt.rc('axes',unicode_minus=False)
plt.rcParams["figure.figsize"] = (20, 9)
plt.rcParams["axes.labelsize"] = 10
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)

# ----- Define function -----
def getBarrel(df, v, min_ang, max_ang):
    return (df.HIT_VEL >= v).values & (df.HIT_VEL < v+5) & (df.HIT_ANG_VER >= min_ang).values & (df.HIT_ANG_VER <= max_ang).values


# ----- Main thread -----
if __name__ == '__main__':
    pStart = time.perf_counter()

    # data load
    origin = pd.read_csv('./data/2021_BigContest_KBO_HTS_Barrel.csv', encoding='CP949', thousands=',')
    
    ret = pd.DataFrame()

    grouped = origin.groupby('velocity')
    for v, df in grouped:
        print(f'[Group {v}km] data count : {len(df)}, ratio : {round(len(df)/len(origin), 4)}')
        if len(df) / len(origin) < 0.1:
            continue
        m = {'velocity': v, 'ang1': -1, 'ang2': 90, 'avg': 1.0, 'slg': 4.0}
        #m = {'velocity': v, 'ang1': 90.0, 'ang2': -1, 'avg': 1.0, 'slg': 4.0}
        #e_avg_slg = 999999
        #e_avg = 999999
        #e_slg = 999999
        #e_diff = 999999
        e_avg_slg = 0
        e_avg = 0
        e_slg = 0
        e_diff = 0
        for _, ang1, ang2, avg, slg in df.values:
            e1 = (avg - 0.5 + slg - 1.5)
            e2 = avg - 0.5
            e3 = slg - 1.5
            e4 = (ang2 - ang1)**2
            #if e_avg_slg > e1 and ang1 > m['ang1'] and ang2 < m['ang2']:
            #if e_avg > e2 and e_slg > e3 and ang1 > m['ang1'] and ang2 < m['ang2']:
            #if e_avg_slg > e1 and e_avg > e2 and e_slg > e3 and ang1 > m['ang1'] and ang2 < m['ang2']:
            #if e_avg_slg > e1 and e_avg > e2 and e_slg > e3:
            #if e_diff > e4:
            #if e_avg_slg < e1 and ang1 > m['ang1'] and ang2 < m['ang2']:
            #if e_avg < e2 and e_slg < e3 and ang1 > m['ang1'] and ang2 < m['ang2']:
            if e_avg_slg < e1 and ang1 > m['ang1']:
            #if e_avg < e2 and e_slg < e3 and ang1 > m['ang1']:
            #if e_avg_slg < e1 and e_avg < e2 and e_slg < e3 and ang1 > m['ang1']:
                e_avg_slg = e1
                e_avg = e2
                e_slg = e3
                e_diff = e4
                m['ang1'] = ang1
                m['ang2'] = ang2
                m['avg'] = avg
                m['slg'] = slg
        ret = ret.append(m, ignore_index=True)

    print(ret)
    if True:
        plt.plot(ret.velocity, ret.ang1, ret.velocity, ret.ang2)
        plt.legend(['ang1', 'ang2'])
        plt.grid()
        plt.xlabel('velocity')
        plt.ylabel('angle')
        plt.show()

    # check graph
    V = []
    a1 = []
    a2 = []
    for v, df in grouped:
        ratio = len(df)/len(origin)
        print(f'[{v} km] ratio = {round(ratio, 4)}')
        if ratio < 0.05:
            continue
        V.append(v)
        a1.append(df.ang1.mode()[0])
        a2.append(df.ang2.mode()[0])
        #a1.append(df.ang1.mean())
        #a2.append(df.ang2.mean())
        #a1.append(df.ang1.quantile(.75))
        #a2.append(df.ang2.quantile(.25))
    d1 = {'x': V, 'y': a1}
    d2 = {'x': V, 'y': a2}
    L1 = LinearRegression()
    L2 = LinearRegression()
    L1.fit(pd.DataFrame(d1['x']), d1['y'])
    L2.fit(pd.DataFrame(d2['x']), d2['y'])
    pred1 = L1.predict(pd.DataFrame(d1['x']))
    pred2 = L2.predict(pd.DataFrame(d2['x']))
    m1 = (pred1[len(pred1)-1] - pred1[0]) / (d1['x'][len(d1['x'])-1] - d1['x'][0])
    n1 = pred1[0] - (m1 * d1['x'][0])
    m2 = (pred2[len(pred2)-1] - pred2[0]) / (d2['x'][len(d2['x'])-1] - d2['x'][0])
    n2 = pred2[0] - (m2 * d2['x'][0])
    for idx, v in enumerate([(m1, n1, d1), (m2, n2, d2)]):
        if v[1] == 0:
            print(f'L{idx+1}, y = {round(v[0], 4)}x')
        elif v[1] > 0:
            print(f'L{idx+1}, y = {round(v[0], 4)}x + {round(v[1], 4)}')
        else:
            print(f'L{idx+1}, y = {round(v[0], 4)}x - {round(abs(v[1]), 4)}')

    if False:
        plt.plot(V, a1, V, pred1, V, a2, V, pred2)
        #plt.plot(V, a1, V, pred1, V, a2, V, pred2, V, ret[ret.velocity.isin(V)].ang1, V, ret[ret.velocity.isin(V)].ang2)
        plt.legend(['ang1 mode', 'ang1 linear-regression', 'ang2 mode', 'ang2 linear-regression'])
        #plt.legend(['ang1 max', 'ang1 linear-regression', 'ang2 min', 'ang2 linear-regression', 'ang1 q3', 'ang2 q1'])
        plt.xlabel('velocity')
        plt.ylabel('angle')
        plt.grid()
        plt.show()

    
    '''
    # barrel labeling
    origin2 = pd.DataFrame()
    for df in [pd.read_csv(f'./data/2021_BigContest_KBO_HTS_{2018+n}.csv', encoding='CP949', thousands=',') for n in range(4)]:
        origin2 = origin2.append(df)
    labels = np.array(['1루타', '2루타', '3루타', '내야안타(1루타)', '땅볼아웃', 
                       '번트아웃', '번트안타', '병살타', '삼중살타', '야수선택', 
                       '인필드플라이', '직선타', '파울플라이', '플라이', '홈런', 
                       '희생번트', '희생플라이'])
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    for el in encoded_labels:
        origin2['HIT_RESULT'].replace(labels[el], el, inplace=True)

    for v, min_ang, max_ang in ret.loc[:, ['velocity', 'min_ang', 'max_ang']].values:
        df = origin2.copy()
        # 각도 범위를 상/하 25% 확장
        df['BARREL'] = getBarrel(df, v, min_ang*0.75, max_ang*1.25)
        plt.hist(df[df.BARREL].HIT_RESULT, bins=range(17), rwidth=0.9, align='left', orientation='horizontal')
        plt.xlabel('frequency')
        plt.title(f'Hist of hit result ({v}km, total: {len(df[df.BARREL])})')
        plt.ylim(-1, 17)
        plt.yticks(list(range(17)), labels, rotation=45)
        #plt.show()
        #plt.savefig(f'hist_hit_result_{v}to{v+5}km.png')
        #df.to_csv(f'./HTS_{v}to{v+5}_Barrel.csv', index=False)
    '''
    
