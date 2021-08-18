import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time, os
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor as PPE

# MLB Barrel 기준(AVG 0.5, SLG 1.5) 만족하는 '타구속도, 발사각도 범위, 타율, 장타율' 계산 함수
# 병렬처리 함수
def work(df, v, values, partition, columns, result):
    for i in range(0, partition):
        for j in range(0, partition):
            a1 = values[i][0]
            a2 = values[j][1]
            df['BARREL'] = getBarrel(df, v, a1, a2)
            b = df[df.BARREL == True]
            avg, slg = getAVG_SLG(b)
            if avg >= 0.5 and slg >= 1.5:
                ret = pd.DataFrame(data=[[v, a1, a2, avg, slg]], columns=columns)
                result.put(ret)

# 타구속도, 발사각도 범위를 이용한 Barrel tagging 함수
def getBarrel(df, v, min_ang, max_ang):
    return (df.HIT_VEL >= v).values & (df.HIT_ANG_VER >= min_ang).values & (df.HIT_ANG_VER <= max_ang).values

# 타율 계산 함수
def getAVG(df):
    #안타 : 1루타, 2루타, 3루타, 내야안타(1루타), 번트안타, 직선타, 홈런
    #타자데이터랑 안타수가 딱 맞진 않음(직선타, 병살타 등에 안타/아웃 나뉘어 있는 듯)
    isHit = [0,1,2,3,6,14]
    H = df[df.HIT_RESULT.isin(isHit)].HIT_RESULT.count()
    AB = df.HIT_RESULT.count()
    return H/AB

# 장타율 계산 함수
def getSLG(df):
    H1 = df[df.HIT_RESULT.isin([0,3,6])].HIT_RESULT.count()
    H2 = df[df.HIT_RESULT == 1].HIT_RESULT.count()
    H3 = df[df.HIT_RESULT == 2].HIT_RESULT.count()
    HR = df[df.HIT_RESULT == 14].HIT_RESULT.count()
    AB = df.HIT_RESULT.count()
    return (H1 + 2*H2 + 3*H3 + 4*HR) / AB

# 타율, 장타율 계산 함수
def getAVG_SLG(df):
    return (getAVG(df), getSLG(df))


# Main thread
if __name__ == '__main__':
    pStart = time.perf_counter()
    
    # 데이터 로드
    file_name = '2021_BigContest_KBO_HTS_2020'
    origin = pd.read_csv(f'./data/{file_name}.csv', encoding='CP949', thousands=',')

    # HIT_RESULT 라벨링
    labels = np.array(['1루타', '2루타', '3루타', '내야안타(1루타)', '땅볼아웃', 
                       '번트아웃', '번트안타', '병살타', '삼중살타', '야수선택', 
                       '인필드플라이', '직선타', '파울플라이', '플라이', '홈런', 
                       '희생번트', '희생플라이'])
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    for el in encoded_labels:
        origin['HIT_RESULT'].replace(labels[el], el, inplace=True)

    # plt 설정
    plt.rc('font', family='Malgun Gothic')
    plt.rc('axes',unicode_minus=False)
    plt.rcParams["figure.figsize"] = (20, 9)
    plt.rcParams["axes.labelsize"] = 10
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    # 분석결과 저장 변수
    col_names = ['velocity', 'min_ang', 'max_ang', 'avg', 'slg']
    barrel_func = pd.DataFrame(columns=col_names)

    # 분석대상 타구속도 (140 ~ 185, 5단위)
    V = np.arange(140, 190, 5)

    # 분석대상 발사각도 (0.0 ~ 89.9, 0.1단위)
    min_ang_range = np.arange(0, 45, 0.1)
    max_ang_range = np.flip(np.arange(45, 90, 0.1))
    ang_ranges = pd.DataFrame(data={'min_ang': min_ang_range, 'max_ang': max_ang_range})

    # 병렬처리 정의
    processes = 10
    partition = int(len(ang_ranges) / processes)
    angs = []
    result = Manager().Queue()
    for i in range(processes):
        angs.append(ang_ranges.iloc[i*partition: (i+1)*partition, :])

    # 분석시작 (병렬처리)
    for i, v in enumerate(V):
        start = time.perf_counter()
        with PPE(processes) as executor:
            _ = [executor.submit(work, origin.copy(), v, angs[i].values, partition, col_names, result) for i in range(processes)] 
        print(f'Parallel#{i} done : {round(time.perf_counter() - start, 4)} secs')
        print(f'- Target velocity : {v}')
    print('')
    
    # 분석결과 저장
    while result.qsize():
        tmp = result.get()
        barrel_func = barrel_func.append(tmp)
    
    print(f'barrel_func.head()\n{barrel_func.head()}')
    print('')
    print(f'barrel_func.describe()\n{barrel_func.describe()}')

    if not os.path.isdir('./data'):
        os.mkdir('./data')
    barrel_func.to_csv(f'./data/{file_name}_Barrel.csv', index=False)

    # 프로그램 런타임
    print(f'Program done : {round(time.perf_counter() - pStart, 4)} secs')
