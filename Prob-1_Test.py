import pandas as pd

# OBP 계산 함수
def getOBP(df):
    # OBP(출루율) = (안타 + 사구)/(사구 + 타수 + 희생플라이)
    TOP = df.HIT + df.HP
    BOTTOM = df.HP + df.AB + df.SF
    return TOP/BOTTOM

# OPS 계산 함수
def getOPS(df):
    # OPS = OBP(출루율) + SLG(장타율)
    OBP = df.OBP
    SLG = df.SLG
    return OBP + SLG

# Main thread
if __name__ == '__main__':

    # 2018년도 타자 클래식 데이터 로드
    file_name = '2021_BigContest_KBO_PlayerRecord_2018'
    dfClassicData = pd.read_csv(f'./data/{file_name}.csv', encoding='CP949', thousands=',')
    # print(dfClassicData.columns)

    dfClassicData['OBP'] = getOBP(dfClassicData)
    dfClassicData['OPS'] = getOPS(dfClassicData)
    # print(dfClassicData)


'''

import seaborn as sns
import matplotlib.pyplot as plt

# Main thread
if __name__ == '__main__':

    # 2018년도 타자 클래식 데이터 로드
    file_name = '2021_BigContest_KBO_PlayerRecord_2018'
    dfClassicData = pd.read_csv(f'./data/{file_name}.csv', encoding='CP949', thousands=',')

    # dfClassicData에 OBP, OPS 계산 후 칼럼 추가
    dfClassicData['OBP'] = (dfClassicData['HIT'] + dfClassicData['BB']) / (
                dfClassicData['BB'] + dfClassicData['AB'] + dfClassicData['SF'])
    dfClassicData['OPS'] = dfClassicData['SLG'] + dfClassicData['OBP']
    # print(dfClassicData)

    # 기록인정 규정타석 수 만족한 선수
    # 규정타석은 경기수*3.1 이다.
    # 현재 프로야구 경기수가 팀당 144경기이니 446.4 소수점이하 버리고(미국,일본은 반올림) 446타석을 채우면 된다.
    # 타석수 > 해당팀 경기수 * 3.1
    # PA 컬럼을 선택합니다. (타석)
    # 컬럼의 값과 조건을 비교합니다.
    # 그 결과를 새로운 변수에 할당합니다.
    # 타석수가 446이상이면서 2018~ 2020년 데이터면 True , 아니면 False??

    temp = (dfClassicData['PA'] >= 446)

    # 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
    subset_df = dfClassicData[temp]

    # 결과를 출력합니다.
    print(subset_df.shape)
    print(subset_df.head())

    # 전체 상관관계 분석 pair plot
    # sns.pairplot(dfClassicData)
    # plt.show()

    # 전체 상관관계 분석 heatmap
    # sns.heatmap(dfClassicData.corr(), annot=True, cmap='YlGnBu',annot_kws={"size": 10})
    # plt.show()


    # Find correlations with the target and sort
    correlations = dfClassicData.corr()['OPS'].sort_values()

    # Display correlations
    print('Most Positive Correlations:\n', correlations.tail(11))
    print('\nMost Negative Correlations:\n', correlations.head(10))

    # 상위 10개 상관관계 bar plot
    # plot 서식 설정
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=75)
    plt.xlabel('Top 10 correlations')
    plt.ylabel('Correlation (%)')
    plt.title('Top 10 correlations with OPS')

    correlations = dfClassicData.corr()['OPS'].sort_values()
    plt.bar(correlations.tail(11)[:-1].index.astype(str)[::-1], 100*correlations.tail(11)[:-1][::-1],color='r')
    # plt.show()

    # 상관관계가 높은 상위 5개 pair plot
    print("Top 5. High correlation with 'OPS' Pair Plot")
    high_corr = dfClassicData.loc[:, list(correlations.tail(6)[::-1].index)]
    sns.pairplot(high_corr, diag_kind='kde')
    plt.show()
    
        # 모든 파일 합치기 
    for n in range(4):
        strHTSPath = '2021_BigContest_KBO_HTS_%d.csv' % (n + 2018)
        strPlayerPath = '2021_BigContest_KBO_PlayerInfo_%d.csv' % (n + 2018)
        strClassicDataPath = '2021_BigContest_KBO_PlayerRecord_%d.csv' % (n + 2018)
        print(strHTSPath)
    
        df = pd.read_csv(strHTSPath, encoding='CP949', thousands=',')
        # df['year'] = (n + 2018)
        dfHTS = pd.concat([dfHTS, df])
    
        df = pd.read_csv(strPlayerPath, encoding='CP949', thousands=',')
        # df['year'] = (n + 2018)
        dfPlayer = pd.concat([dfPlayer, df])
    
        df = pd.read_csv(strClassicDataPath, encoding='CP949', thousands=',')
        # df['year'] = (n + 2018)
        dfClassicData = pd.concat([dfClassicData, df])
        
        # print(result1)    
        strTeamDataPath = '2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_팀.csv'
        dfTeamData = pd.read_csv(strTeamDataPath, encoding='CP949', thousands=',')
        
        strMatchDatePath = '2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_경기일정_2021.csv'
        dfMatchDateData = pd.read_csv(strMatchDatePath, encoding='CP949', thousands=',')
        
        print(dfTeamData.shape)
        print(dfMatchDateData.shape)
        
        import numpy as np
        grouped = dfHTS.groupby(dfHTS['GYEAR']) # 년별로 그룹핑 
        #dfHTS['PCODE']
        it=iter(grouped)
        gamesinHTS = []
        for n in range(4):
            shipNo, dFrame = next(it) # 2018 
            print(shipNo)
            playerGames = dFrame.groupby(dFrame['PCODE']) # 선수별 구룹핑  
            # HTS 데이터 선수별 경기수 카운트
        
            #리스트 화 
            # 선수, G_ID 중복 제거하여 unique한 것 리스트 
            # 년별로 경기수 산출 
            PA= playerGames['G_ID'].unique().reset_index() #  선수가 참여한 게임 의 고유값 리스트 추출 
            print(PA.shape)
        
            lstCount = []
            # HTS에서 G_ID 고유값 리스트 count 값을 경기수로 계산.
            for n in PA['G_ID']:
                lstCount.append(len(n))
            dfd = pd.DataFrame(lstCount, columns=['Count'])
            PA = pd.concat([PA, dfd], axis = 1)
            gamesinHTS.append(PA)
'''





