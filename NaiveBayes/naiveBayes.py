import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import comb
from stqdm import stqdm
from time import sleep

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from DatasetINFO.transposeDataset import transpose as ts

def naive_bayes(iris_pd):
    print('start naive bayes ---------------------------------')
    st.header("What is Naive Bayes Clisifi? ::", divider='blue')
    
    st.subheader("Gaussian Naive Bayes")


    st.header("Split Dataset :scissors:", divider='blue')
    
    # split train, test
    variety=iris_pd.variety
    data=iris_pd.drop(columns='variety')    # 순수 데이터값만 출력

    iris_pd['category']=iris_pd['variety'].factorize()[0]   # class를 .factorize()[0]로 숫자로 치환하여 'category' 컬럼에 추가
    category=iris_pd['category']

    # st.markdown("##### 데이터셋 분할 및 class 순 정렬")
    st.text("데이터셋 분할 비율 그래프 나타내기")
    st.text("데이터셋 분할 옵션")

    # 분할한 train 데이터셋
    data_train, data_test, variety_train, variety_test=train_test_split(
        data, category, test_size=0.2, stratify=category, random_state=1
        )
    # st.dataframe(data_train)
    print('variety bincount : ',np.bincount(variety_train))     # 빈도수 세기
    newIris=pd.DataFrame(np.column_stack([data_train, variety_train]))

    # sort and rearrange the data based on the variety : 분할한 데이터셋을 클래스 순서로 정렬
    setosa=newIris[newIris[4]==0]
    versicolor=newIris[newIris[4]==1]
    virginica=newIris[newIris[4]==2]
    newIris=pd.concat([setosa, versicolor, virginica])
    newIris_df = pd.DataFrame({
        'sepal.length':newIris[0], 
        'sepal.width':newIris[1], 
        'petal.length':newIris[2], 
        'petal.width':newIris[3],
        'variety':newIris[4]
        })
    st.dataframe(newIris_df)


    st.header("Training Model ::", divider='blue')
    # split data based on variety
    setosa_data=newIris[0:40]
    versicolor_data=newIris[40:80]
    virginica_data=newIris[80:120]

    # find mean
    setosa_mean=setosa_data.mean()
    versicolor_mean=versicolor_data.mean()
    virginica_mean=virginica_data.mean()

    # find standard diviation
    setosa_std=setosa_data.std()
    versicolor_std=versicolor_data.std()
    virginica_std=virginica_data.std()


    # 학습 데이터 셋에서 우도 찾기
    st.markdown('train likeilhood')
    print('finding train likelihood')
    x=[]
    likelihood=[]

    for i in range(len(newIris)):        # stqdm 완전 끝나면 로딩바 자체가 사라짐
        # sleep(0.5)
        distribution=1
        if(i<40):                               #setosa
            mean=setosa_mean
            std=setosa_std
        if(i>=40 and i<80):                     #versicolor
            mean=versicolor_mean
            std=versicolor_std
        if(i>=80 and i < 120):                  #virginica
            mean=virginica_mean
            std=virginica_std
        
        for j in range(4):  # 우도 구하기
            x = newIris.iloc[i]
            a = ((x[j]-mean[j])**2)/(2*std[j]**2)
            b = math.sqrt(2*math.pi*(std[j]**2))
            y = math.exp(-a)/b
            distribution=distribution*y
        likelihood.append(distribution) # 우도 값 넣기
        x=[]
    st.dataframe(likelihood)

    # find priori probability = 사전확률. 해당 클래스일 확률. 50/150
    setosa_priori = len(setosa_data)/len(newIris)
    versicolor_priori = len(versicolor_data)/len(newIris)
    virginica_priori = len(virginica_data)/len(newIris)

    print('setosa_priori :', setosa_priori)
    print('versicolor_priori :', versicolor_priori)
    print('virginica_priori :', virginica_priori)
    print()

    st.header("Testing Model ::", divider='blue')
    # rearrange the data into groups based on the variety
    print('test_data start-------------------------------')
    newTest = pd.DataFrame(np.column_stack([data_test, variety_test]))
    setosa=newTest[newTest[4]==0]
    versicolor=newTest[newTest[4]==1]
    virginica=newTest[newTest[4]==2]
    newTest=pd.concat([setosa, versicolor, virginica])
    st.dataframe(newTest)
    
    # test데이터셋에서 우도 찾기
    print('finding test likelihood')
    x=[]
    testLikelihood=[]
    testPosterior=[]
    posteriorVariety=[]

    for i in range(len(newTest)):
        for c in range(3):
            if(c==0):
                mean=setosa_mean
                std=setosa_std
                priori=setosa_priori
            if(c==1):
                mean=versicolor_mean
                std=versicolor_std
                priori=versicolor_priori
            if(c==2):
                mean=virginica_mean
                std=virginica_std
                priori=virginica_priori
            distribution=1

            for j in range(4):
                x=newTest.iloc[i]
                a=((x[j]-mean[j])**2)/(2*std[j]**2)
                b=math.sqrt(2*math.pi*(std[j]**2))
                y=math.exp(-a)/b
                distribution=distribution*y
            x=[]
            testLikelihood.append(distribution)
            posterior=testLikelihood[c]*priori          # Calculate poterior values = 사후확률 계산
            testPosterior.append(posterior)
            maxPosterior=testPosterior.index(max(testPosterior))
        posteriorVariety.append(maxPosterior)

        testLikelihood=[]
        testPosterior=[]
    st.markdown('Naive Bayes로 예측한 클래스')
    posterior_df=pd.DataFrame({'posterior':posteriorVariety})
    st.dataframe(posterior_df.groupby('posterior')['posterior'].count())

    # check the differences
    print("variety of original test data")
    variety_test=list(map(int, newTest[4]))
    print(variety_test)

    print("variety of Maximum Posterior values")
    print(posteriorVariety)
    print()

    # Accuarcy
    similar=0
    for i in range(len(posteriorVariety)):
        if(variety_test[i]==posteriorVariety[i]):
            similar += 1
    accuarcy=similar/(i+1)*100

    print('accuarcy : ',accuarcy)

    # fig, ax = plt.subplots()
    # for i in range(len(posteriorVariety)):
    #     ax.plot(variety_test[i])
    #     ax.plot(posteriorVariety[i])

    cm = confusion_matrix(posteriorVariety, variety_test)
    print('confusion matrix')
    print(cm)
    plot = sns.heatmap(
        cm, 
        annot=True, # 히트맵 각 항목의 숫자 표시 여부
        cmap='BuPu' # 색 테마 선택
    )
    plot.set(xlabel="test", ylabel="true")
    st.pyplot(plot.get_figure())
