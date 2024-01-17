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
    # st.header("What is Naive Bayes? :speech_balloon:", divider='blue')
    # st.text("1.간단 예시 설명+Bayes 유래+그림자료 필요")
    st.text_area("Overview",
                 "Naive Bayes는 유명한 분류 알고리즘으로 확률 이론의 베이즈 정리를 기반으로 만들어졌어요. "
                 "때문에 이름에도 Bayes가 들어가 있습니다. "
                 "이 알고리즘은 스팸 메일 분류부터 시작하여 감정분석 등 광범위하게 사용돼요.  "
                 "대표적으로 넷플릭스의 추천 알고리즘도 이 알고리즘이 사용됐어요. "
                 )
    st.image('image/naivebayes.jpg')
    
    # st.markdown('''
    #             ##### 확률
    #             어떠한 사건 A가 일어날 확률을 표기할 때 **P(A)** 라 해요.
    #             사건A를 주사위를 던졌을 때 3이 나오는 사건이라면 1/6의 확률을 가지기 때문에
    #             아래와 같이 표기해요.
    #             ''')
    # st.latex(r'''P(A)=\frac{1}{6}''')
    # st.markdown('''
    #             ##### 조건부 확률
    #             Naive Bayes는 **조건부 확률**을 이용해 데이터를 분류해요. 
    #             '비가 올때 튀김류 요리 매출이 오를 확률' 또는 '주사위 두개 중 하나가 3이 나온 후 다른 하나가 2가 나올 확률'과 같이 
    #             사건 A가 일어난 후 사건 B가 일어날 확률을 의미하며 아래와 같이 표기해요.
    #             ''')
    # st.latex(r'''P(B|A)''')

    # 각 요소 상세 설명
    st.latex(r'''사후확률=\frac{우도\times 사전확률}{주변 우도}''')
    st.markdown('''
                ###### P(A):사전확률(Priori)
                사후확률을 구할 때 사건 B가 일어나기 전의 확률이라서 사전 확률이라고 해요.  
                다시 말해 데이터가 어디로 분류될지 즉, 어느 특정 class 하나로 data가 분류될 확률이에요.  
                Iris dataset으로는 아래와 같이 표기할 수 있어요.
                ''')
    st.latex(r'''
            \frac{클래스별 데이터 개수}{전체 데이터 개수}=\frac{200}{600}=\frac{1}{3}=P(class)
            ''')
    st.markdown('''
                ###### P(B|A):우도함수(likelihood)
                가능도라 부르기도 하며 우리가 이미 알고 있는 관측값에 대해 
                실제값과 얼마나 일관되는지 그 정도를 구하는 함수를 말해요.  
                우리가 가지고 있는 dataset을 통해 분류할 데이터가 어떤 class의 값일지 class를 고정해두고 
                확률을 구해보는 것과 비슷해요.
                ''')
    st.latex(r'''
            P(data|class)
            ''')
    st.markdown('''            
                ###### P(A|B):사후확률(evidence)
                사건 A가 일어난 후에 사건 B가 일어날 확률을 말해요. 
                Naive Bayes의 분류 기준이자 얼마나 일관성 있게 판단했는지를 나타내요. 
                사후 확률은 데이터를 분류할 때, 분류할 목록들 각각 구해서 비교 후 가장 높은 항목으로 분류해요.
                ''')
    st.latex(r'''
            P(class|data)
            ''')
    # 왜 Naive 인지
    st.markdown('''
                Naive Bayes는 데이터를 분류할 때 영향을 미치는 모든 조건들이 독립적이라고 가정해요.
                실제로는 다양한 조건들이 서로 연관 되어 있지만 이를 단순하게 여기고 계산한다고 
                순진(Naive)하다고 해서 앞에 Naive가 붙었어요. 
                ''')
    # 특징 나열 + 장단점
    st.markdown('''
                ##### 장점  
                + 빠르고 간단하여 효율적임  
                    : 단순한 구조로 되어 있으며 입력받는 데이터 종류 사이의 독립성을 가정하고 있어서 
                    모델을 이해하고 만들기 쉬워요.    
                 
                + 신뢰도 제공  
                    : Naive Bayes는 사후확률을 구하여 분류하는데 이 값은 모델이 예측값에 
                    대해 확신하는 정도를 보여줘요. 이를 활용하여 모델이 에측을 잘 수행하는지를 표현하는 
                    중요한 지표로 활용 될 수 있어요.
                + 데이터 크기에 상관 없이 잘 작동함  
                    : 모델이 간단하고 독립성을 가정하기 때문에, 다른 복잡한 모델들보다 적은 양의 훈련 데이터로도 좋은 성능을 낼 수 있어요.
                + 이진 및 다중 분류에도 사용 가능  
                    : 다양한 상황에 유연하게 적용될 수 있어요.
                + 노이즈와 누락 데이터를 잘 처리
                  
                ##### 단점  
                - 모든 특징이 동등하게 중요  
                    : 데이터 특징들 간의 가중치를 고려하지 않아서 실제 중요한 데이터 또는 중요하지 않는 데이터도 동등하게 다뤄 
                    정확하지 못한 예측을 할 수 있어요.
                - 독립이라는 가정이 잘못된 경우 
                    : 현실에서는 변수들이 항상 독립적이지 않을 수 있습니다. 이 가정이 성립하지 않는 경우 
                    모델의 성능이 저하될 수 있어요.
                - 0 확률
                    : 훈련 데이터에서 특정 클래스에 대한 특정 데이터의 값이 전혀 나타나지 않으면, 해당 변수에 대한
                    확률이 0이 되어 모델이 예측할 때 문제를 일으킬 수 있어요.
                - 선형 분리 불가능
                    : 클래스들을 구분할 때 복잡한 경계를 가지면 성능이 저하될 수 있어요.
                ''')
    st.markdown('''
                #### Guassian Naive Bayes
                ''')
    st.image('image/gaussian.jpg')
    st.markdown('''
                Naive Bayes의 여러 종류 중 Gaussian Naive Bayes는 데이터가 iris 데이터와 같이 
                연속적인 수로 수량화가 가능한 데이터일 때 정규 분포를 따른다는 가정에서 적용해요.
                ''')


    st.header("Workflow", divider='blue')
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
    st.text('Gaussian Naive Bayes - likeilhood')
    st.latex(r'''
            P(train|class)=f(x_i|y)=\frac{1}{\sqrt{2 \pi \sigma_y^2}}
             \exp(-\frac{(x_i- \mu_y)^2}{2 \sigma_y^2})
            ''')
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
    #가능하면 사후확률 구하기
    
    setosa_likelihood=likelihood[0:40]
    versicolor_likelihood=likelihood[40:80]
    virginica_likelihood=likelihood[80:120]
    train_likelihood=pd.DataFrame({
        'Setosa':setosa_likelihood, 
        'Versicolor':versicolor_likelihood, 
        'Virginica':virginica_likelihood})
    st.dataframe(train_likelihood)      # 클래스별로 column 재구성해서 보여주기
    
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
    
    st.header("Model evaluation", divider='blue')

    # Accuarcy
    similar=0
    for i in range(len(posteriorVariety)):
        if(variety_test[i]==posteriorVariety[i]):
            similar += 1
    accuarcy=similar/(i+1)*100

    # 정확도, f1 score 묶어서 보여주기
    print('accuarcy : ',accuarcy)
    # macro : 각 라벨에 대한 f1-score 평균값. 모든 라벨 동등 조건으로 보는 naive bayes라 선택
    f1 = f1_score(posteriorVariety, variety_test, average='macro')
    print('f1_score : ', f1)
    st.dataframe({
        'accuarcy':accuarcy,
        'f1_score':f1
    })

    # fig, ax = plt.subplots()
    # for i in range(len(posteriorVariety)):
    #     ax.plot(variety_test[i])
    #     ax.plot(posteriorVariety[i])
    fig1, ax1=plt.subplots(figsize=(3,2.5))
    cm = confusion_matrix(posteriorVariety, variety_test)
    print('confusion matrix')
    print(cm)
    ax1 = sns.heatmap(
        cm, 
        annot=True, # 히트맵 각 항목의 숫자 표시 여부
        cmap='BuPu', # 색 테마 선택
    )
    ax1.set_title("Confusion matrix")
    ax1.set(xlabel="test", ylabel="true")
    st.pyplot(fig1)
