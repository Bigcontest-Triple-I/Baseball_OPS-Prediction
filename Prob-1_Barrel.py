import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time, os
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor as PPE

import warnings
warnings.filterwarnings('ignore')

# ----- Global variable -----
# Target velocity
vel_limit = 190
vel_unit = 5
V = np.arange(150, vel_limit+0.1, vel_unit)

# Target angle pattern
ang_ranges = []
ang_limit = 80
ang_unit = 1
for i in np.arange(0, ang_limit, ang_unit):
    for j in np.arange(i+ang_unit, ang_limit, ang_unit):
        ang_ranges.append({'ang1': i, 'ang2': j})

# ----- Define function -----
# Parallel processing function
def parallel(df, v, values, columns, result):
    # MLB Barrel Rules : AVG 0.5, SLG 1.5
    for ang in values:
        a1 = ang['ang1']
        a2 = ang['ang2']
        df['BARREL'] = getBarrel(df, v, a1, a2)
        #if len(df[df.BARREL])/len(df) < 0.001:
        #if len(df[df.BARREL]) < 10:
            #continue
        avg, slg = getAVG_SLG(df[df.BARREL])
        if avg >= 0.5 and slg >= 1.5:
            data = [v, a1, a2, avg, slg]
            ret = pd.DataFrame([data], columns=columns)
            result.put(ret)

# Decide a hit is 'BARREL' or not
def getBarrel(df, v, min_ang, max_ang):
    return (df.HIT_VEL >= v).values & (df.HIT_VEL < v + vel_unit) & (df.HIT_ANG_VER >= min_ang).values & (df.HIT_ANG_VER <= max_ang).values

# Calculate AVG
def getAVG(df):
    #타자데이터랑 안타수가 딱 맞진 않음(직선타, 병살타 등에 안타/아웃 나뉘어 있는 듯)
    #isHit = [0,1,2,3,6,14]
    isHit = [0,1,2,3,14]
    H = df[df.HIT_RESULT.isin(isHit)].HIT_RESULT.count()
    AB = df.HIT_RESULT.count()
    return H / AB

# Calculate SLG
def getSLG(df):
    H1 = df[df.HIT_RESULT.isin([0,3,6])].HIT_RESULT.count()
    H2 = df[df.HIT_RESULT == 1].HIT_RESULT.count()
    H3 = df[df.HIT_RESULT == 2].HIT_RESULT.count()
    HR = df[df.HIT_RESULT == 14].HIT_RESULT.count()
    AB = df.HIT_RESULT.count()
    return (H1 + 2*H2 + 3*H3 + 4*HR) / AB

# Calculate AVG and SLG
def getAVG_SLG(df):
    return (getAVG(df), getSLG(df))


# ----- Main thread -----
if __name__ == '__main__':
    pStart = time.perf_counter()

    origin = pd.DataFrame()
    for n in range(4):
        origin = origin.append(pd.read_csv(f'./data/2021_BigContest_KBO_HTS_{2018+n}.csv', encoding='CP949', thousands=','))

    # HIT_RESULT labeling
    labels = np.array(['1루타', '2루타', '3루타', '내야안타(1루타)', '땅볼아웃', 
                       '번트아웃', '번트안타', '병살타', '삼중살타', '야수선택', 
                       '인필드플라이', '직선타', '파울플라이', '플라이', '홈런', 
                       '희생번트', '희생플라이'])
    encoder = LabelEncoder()
    for label in encoder.fit_transform(labels):
        origin['HIT_RESULT'].replace(labels[label], label, inplace=True)
    print(f'HTS data count = {len(origin)}')

    # HIT_RESULT filtering
    filtering = False
    if filtering:
        '''
        filters = np.array(['1루타', '2루타', '3루타', '병살타', '삼중살타', 
                           '직선타', '파울플라이', '플라이', '홈런', '희생플라이'])
        '''
        #filters = np.array([0, 1, 2, 7, 8, 11, 12, 13, 14, 16])
        '''
        filters = np.array(['1루타', '2루타', '3루타', 
                           '직선타', '파울플라이', '플라이', '홈런', '희생플라이'])
        '''
        filters = np.array([0, 1, 2, 11, 12, 13, 14, 16])
        tmp = origin[origin.HIT_RESULT.isin(filters)]
        print(f'Sample data count = {len(tmp)}')
        print(f'Sample data ratio = {round(len(tmp)/len(origin), 4)}')
        origin = tmp.copy()


    '''
    # 분석대상 타구속도 (140 ~ 190, 5단위)
    V = np.arange(150, 191, 10)

    # 분석대상 발사각도 (0 ~ 80, 1단위)
    ang_ranges = []
    ang_unit = 1
    for i in np.arange(0, 80, ang_unit):
        for j in np.arange(i+ang_unit, 80, ang_unit):
            ang_ranges.append({'ang1': i, 'ang2': j})
    '''
    print(f'Target pattern count = {len(ang_ranges) * len(V)}')

    # Setup parallel processing
    processes = 10
    result = Manager().Queue()
    angs = np.array_split(ang_ranges, processes)
    
    if True:
        # Analyze (Parallel)
        columns = ['velocity', 'ang1', 'ang2', 'avg', 'slg']
        barrel_condition = pd.DataFrame(columns=columns)
        print(f'Start parallel process')
        for i, v in enumerate(V):
            start = time.perf_counter()
            executor = PPE(processes)
            _ = [executor.submit(parallel, origin.copy(), v, ang, columns, result) for ang in angs]
            executor.shutdown()
            print(f'Parallel#{i}, {v} km : {round(time.perf_counter() - start, 4)} secs')
        print('')

        print('Merging results...')
        while result.qsize():
            barrel_condition = barrel_condition.append(result.get())
        print(f'Barrel data count = {len(barrel_condition)}')
        print(f'Barrel data ratio = {round(len(barrel_condition)/(len(ang_ranges)*len(V)), 4)}')

        print('')
        print(f'barrel_condition.describe()\n{barrel_condition.describe()}')
        print('')

        if not os.path.isdir('./data'):
            os.mkdir('./data')
        barrel_condition.to_csv(f'./data/2021_BigContest_KBO_HTS_Barrel.csv', index=False)

    # Program runtime
    print(f'Program done : {round(time.perf_counter() - pStart, 4)} secs')
