import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from DatasetINFO.transposeDataset import transpose as ts

def naive_bayes(iris_pd):
    print('start naive bayes ---------------------------------')
    st.header("What is Naive Bayes Clisifi? :speech_balloon:", divider='blue')
    st.text("1.간단 예시 설명+Bayes 유래+그림자료 필요")
    st.text("2.특징 나열 + 장단점")
    st.text("3.각 요소 상세 설명")
    st.text("4.왜 Naive 인지")
    st.text("5.종류 간략 설명")
    st.text("6.가우시안 간단 설명")
    st.latex(r'''P(A|B)=\frac{P(B|A)P(A)}{P(B)}''')

    st.subheader("Gaussian Naive Bayes")


    st.header("Workflow")
    st.image('image/workflow_eng.jpg')
    st.latex(r'''P(class|data)=\frac{P(data|class)\times P(class)}{P(data)}''')

    st.header("Split Dataset :scissors:", divider='blue')

    # 데이터셋 분할 및 class 순 정렬
    st.subheader("Ratio")
    fig, ax = plt.subplots(figsize=(9.2, 1))
    ax.invert_yaxis()
    ax.yaxis.set_visible(False)
    ax.set_xlim(-0.1, 10.1) # 막대그래프 좌우 공백
    ax.barh(
        '0',[8, 2], 
        left=[0, 8],
        label=['train', 'test'],
        color=['mediumpurple','rebeccapurple'], alpha=0.8
        )
    ax.text(
        4, 0, 'train',
        ha='center', va='center',
        color='white',
        size='15'
    )
    ax.text(
        9, 0, 'test',
        ha='center', va='center',
        color='white',
        size='15'
    )
    st.pyplot(fig)


    # split train, test
    variety=iris_pd.variety
    data=iris_pd.drop(columns='variety')    # 순수 데이터값만 출력

    iris_pd['category']=iris_pd['variety'].factorize()[0]   # class를 .factorize()[0]로 숫자로 치환하여 'category' 컬럼에 추가
    category=iris_pd['category']


    st.subheader("Option")
    split_option={
        'x':'data value',
        'y':'variety column',
        'valdation_set': '0%',
        'test_set':'20%',
        'suffle':'True',
        'stratify':'variety'
        }
    st.dataframe(split_option)

    st.subheader("Result")
    st.info("When performing the above operation, replace the variable column with a number.")
    # 분할한 train 데이터셋
    data_train, data_test, variety_train, variety_test=train_test_split(
        data, category, test_size=0.2, stratify=category, random_state=1
        )
    # st.dataframe(data_train)
    print('variety bincount : ',np.bincount(variety_train))     # 빈도수 세기
    newIris=pd.DataFrame(np.column_stack([data_train, variety_train]))

    fig, ax=plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 135)
    ax.xaxis.set_visible(False)
    ax.barh(
        ['train', 'test'],[len(data_train), len(data_test)],
        color=['mediumpurple','rebeccapurple'], alpha=0.8
        )
    ax.text(
        127, 0, str(len(data_train)),
        ha='center', va='center',
        color='black',
        size='15'
    )
    ax.text(
        35, 1, str(len(data_test)),
        ha='center', va='center',
        color='black',
        size='15'
    )
    st.pyplot(fig)
    
    on_result=st.toggle('View all Split Result')
    if on_result:
        train1, train2 = st.columns((2, 1))
        with train1:
            st.dataframe(data_train)
            st.dataframe(data_test)
        with train2:
            st.dataframe(variety_train)
            st.dataframe(variety_test)


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
    

    st.header("Training Model :steam_locomotive:", divider='blue')
    st.subheader('train dataset')
    st.dataframe(newIris_df.head())
    on_train = st.toggle('View all Train dataset')
    if on_train:
        st.dataframe(newIris_df)
    # split data based on variety
    setosa_data=newIris[0:40]
    versicolor_data=newIris[40:80]
    virginica_data=newIris[80:120]
    
    st.subheader('find train dataset mean, standard deviation')
    # find mean
    setosa_mean=setosa_data.mean()
    versicolor_mean=versicolor_data.mean()
    virginica_mean=virginica_data.mean()

    # find standard diviation
    setosa_std=setosa_data.std()
    versicolor_std=versicolor_data.std()
    virginica_std=virginica_data.std()


    # 학습 데이터 셋에서 우도 찾기
    st.subheader('find train likeilhood')
    st.latex(r'''P(train|class)''')
    st.code('''
            a = ((x[j]-mean[j])**2)/(2*std[j]**2)       # x :train data
            b = math.sqrt(2*math.pi*(std[j]**2))
            y = math.exp(-a)/b
            ''', 
            language='python')
    likelihood=[]
    posterior=[]
    posteriorVariety=[]

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
    st.text("클래스별로 column 재구성해서 보여주기-현상황 최선")
    st.text("가능하면 사후확률 구하기")
    st.dataframe(likelihood)
    on_train=st.toggle("Veiw training code")
    code_train='''
        likelihood=[]

        for i in range(len(newIris)):
            # sleep(0.5)
            distribution=1
            if(i<40):                                        #setosa
                mean=setosa_mean
                std=setosa_std
            if(i>=40 and i<80):                              #versicolor
                mean=versicolor_mean
                std=versicolor_std
            if(i>=80 and i < 120):                          #virginica
                mean=virginica_mean
                std=virginica_std
            
            for j in range(4):                              # find likelihood
                x = newIris.iloc[i]
                a = ((x[j]-mean[j])**2)/(2*std[j]**2)
                b = math.sqrt(2*math.pi*(std[j]**2))
                y = math.exp(-a)/b
                distribution=distribution*y
            likelihood.append(distribution)
            x=[]
    '''


    st.subheader('find train priori')
    # find priori probability = 사전확률. 해당 클래스일 확률. 50/150
    setosa_priori = len(setosa_data)/len(newIris)
    versicolor_priori = len(versicolor_data)/len(newIris)
    virginica_priori = len(virginica_data)/len(newIris)

    print('setosa_priori :', setosa_priori)
    print('versicolor_priori :', versicolor_priori)
    print('virginica_priori :', virginica_priori)

    
    if on_train:
        st.code(code_train, language='python')
    print()
    
    st.header("Testing Model :pencil:", divider='blue')
    # rearrange the data into groups based on the variety
    print('test_data start-------------------------------')
    newTest = pd.DataFrame(np.column_stack([data_test, variety_test]))
    setosa=newTest[newTest[4]==0]
    versicolor=newTest[newTest[4]==1]
    virginica=newTest[newTest[4]==2]
    newTest=pd.concat([setosa, versicolor, virginica])

    st.subheader('test dataset')
    st.dataframe(newTest.head())
    on_test = st.toggle('View all Test dataset')
    if on_test:
        st.dataframe(newTest)
    
    # test데이터셋에서 우도 찾기
    print('finding test likelihood')

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
    
    st.subheader('Naive Bayes로 예측한 클래스')
    posterior_df=pd.DataFrame({'posterior':posteriorVariety})
    posterior_df=pd.DataFrame(posterior_df.groupby('posterior')['posterior'].count())
    posterior_df['variety']=['Setosa', 'Versicolor', 'Virginica']
    posterior_df=posterior_df[['variety', 'posterior']]
    st.dataframe(posterior_df)

    on_test=st.toggle("Veiw testing code")
    code_test='''
        testLikelihood=[]
        testPosterior=[]
        posteriorVariety=[]
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
        '''
    if on_test:
        st.code(code_test, language='python')

    # check the differences
    print("variety of original test data")
    variety_test=list(map(int, newTest[4]))
    print(variety_test)

    print("variety of Maximum Posterior values")
    print(posteriorVariety)
    print()
    
    st.subheader("Performance evaluation")
    st.text("정확도, f1 score 묶어서 보여주기")

    # Accuarcy
    similar=0
    for i in range(len(posteriorVariety)):
        if(variety_test[i]==posteriorVariety[i]):
            similar += 1
    accuarcy=similar/(i+1)*100

    print('accuarcy : ',accuarcy)
    # print('f1_score : ', f1_score())

    # fig, ax = plt.subplots()
    # for i in range(len(posteriorVariety)):
    #     ax.plot(variety_test[i])
    #     ax.plot(posteriorVariety[i])
    fig1, ax1=plt.subplots(figsize=(5,4))
    cm = confusion_matrix(posteriorVariety, variety_test)
    print('confusion matrix')
    print(cm)
    ax1 = sns.heatmap(
        cm, 
        annot=True, # 히트맵 각 항목의 숫자 표시 여부
        cmap='BuPu', # 색 테마 선택
    )
    ax1.set(xlabel="test", ylabel="true")
    st.pyplot(fig1)
