import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time, os

from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor as PPE

#import warnings
#warnings.filterwarnings('ignore')

# ----- Defined function -----
def getAVG(df):
    if len(df) == 0:
        return -1
    # 1루타, 2루타, 3루타, 번트안타, 홈런
    isHit = [0,1,2,3,14]
    H = len(df[df.HIT_RESULT.isin(isHit)])
    AB = len(df)
    return H / AB

def getSLG(df):
    H1 = len(df[df.HIT_RESULT.isin([0,3,6])])
    H2 = len(df[df.HIT_RESULT == 1])
    H3 = len(df[df.HIT_RESULT == 2])
    HR = len(df[df.HIT_RESULT == 14])
    AB = len(df)
    return (H1 + 2*H2 + 3*H3 + 4*HR) / AB

# ----- Main thread -----
if __name__ == '__main__':
    pStart = time.perf_counter()

    # Load HTS dataset
    origin = pd.DataFrame()
    for n in range(4):
        origin = origin.append(pd.read_csv(f'./data/2021_BigContest_KBO_HTS_{2018 + n}.csv', encoding='CP949', thousands=','))
    print(f'HTS data count = {len(origin)}')

    # 'HIT_RESULT' labeling
    labels = np.array(['1루타', '2루타', '3루타', '내야안타(1루타)', '땅볼아웃', 
                       '번트아웃', '번트안타', '병살타', '삼중살타', '야수선택', 
                       '인필드플라이', '직선타', '파울플라이', '플라이', '홈런', 
                       '희생번트', '희생플라이'])
    encoder = LabelEncoder()
    for label in encoder.fit_transform(labels):
        origin['HIT_RESULT'].replace(labels[label], label, inplace=True)

    # Data preprocessing
    IQR = origin.quantile(.75) - origin.quantile(.25)
    # Select HIT_ANG_VER between(-40, 80) : most are within 1.5*IQR
    dfHTS = origin[origin.HIT_ANG_VER.between(-40, 80)].copy()
    # Select HIT_VEL within 1.5*IQR
    dfHTS = dfHTS[dfHTS.HIT_VEL.between(dfHTS.HIT_VEL.quantile(.25) - (1.5*IQR.HIT_VEL),
                                        dfHTS.HIT_VEL.quantile(.75) + (1.5*IQR.HIT_VEL))]

    # Drawing AVG, SLG, AVG+SLG per angle-range
    # angle-range unit : 4
    ang_range = np.arange(-40, 80, 4)
    AVG = []
    SLG = []
    AVG_SLG = []
    VEL_MEAN = []
    for a in ang_range:
        df = dfHTS[dfHTS.HIT_ANG_VER.between(a, a+4, 'left')]
        AVG.append(getAVG(df))
        SLG.append(getSLG(df))
        AVG_SLG.append(getAVG(df) + getSLG(df))
        VEL_MEAN.append(df.HIT_VEL.mean())
    fig = plt.figure(figsize=(20,6))
    axes = [fig.add_subplot(1,4,i) for i in range(1, 5)]
    [axes[i].plot(ang_range, data) for i, data in enumerate([AVG, SLG, AVG_SLG, VEL_MEAN])]
    fig.suptitle('AVG, SLG, AVG+SLG, VEL MEAN per Launch angle', fontsize=16)
    [ax.set_xlabel('Launch angle (unit: 4)') for ax in axes]
    [axes[i].set_ylabel(label) for i, label in enumerate(['AVG', 'SLG', 'AVG+SLG', 'VEL MEAN'])]
    [ax.grid() for ax in axes]
    [ax.axvline(0, color='r') for ax in axes]
    [ax.axvline(40, color='r') for ax in axes]
    plt.tight_layout()
    plt.show()
