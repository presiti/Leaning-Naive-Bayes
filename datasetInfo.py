import streamlit as st
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

import normalChart as nc

def dataset_info(iris_pd):
    # dataset source
    st.markdown('Dataset Sources : https://gist.github.com/netj/8836201')


    # Dataset Info
    print('show dataset info\n')

    #show dataset head(sample)
    st.header('iris :violet[sample]', divider='violet')
    st.dataframe(data=iris_pd.head())

    #show dataset info
    st.header('iris :violet[info]', divider='violet')

    col1, col2 = st.columns([1, 2], gap="small")
    with col1:  #show dataset class info
        st.subheader("class info")
        iris_class = iris_pd.groupby('variety')['variety'].count()
        st.dataframe(data=iris_class)
        st.subheader("null check")
        st.dataframe(iris_pd.isnull().sum())

    with col2:  #show dataset describe
        st.subheader('iris describe')
        # print(iris_pd.describe())
        st.dataframe(iris_pd.describe())

    # Graph
    c_list = ['Setosa', 'Versicolor', 'Virginica']                          #class name list
    f_list = [iris_pd.columns[i] for i in range(4)]                         #feature name list

    #show Scatter chart
    st.subheader('iris :blue[scatter chart]', divider='blue')

    iris_trans=pd.DataFrame()

    # 데이터셋 변환
    print('transpose dataset-------------------------')
    print('> transpose dataset \n')
    for i in range(3):              # 0~2, class 수만큼 돌기
        print('transpose :', c_list[i])
        iris_temp = iris_pd[iris_pd.variety==c_list[i]].transpose()     # 1. class별로 떼어와서 transpose로 행과 열을 바꾸기
        iris_np=np.empty(200)                                           # class별 데이터를 담을 numpy 리스트
        
        for j in range(4):          # 0~3, feature 수만큼 돌기
            f_np=iris_temp.iloc[j].to_numpy()                                      # 2. feature 4개를 한 줄씩 떼오기
            print(f_list[j], ' first value : ', f_np[0])
            for k in range(50):
                iris_np[k+j*50]=f_np[k]                                 # 3. 값을 하나씩 꺼내어 넣어 clsas별 numpy 리스트로 만들기
        iris_trans[c_list[i]]=iris_np                                   # 4. 데이터 프레임에 컬럼명을 class명으로 정하여 numpy리스트 넣기
        print(c_list[i], 'done\n')
    print('> add feature column')
    iris_f=list()                                                       # feature 열 값을 담을 리스트
    for j in range(4):
        for k in range(50):
            iris_f.append(f_list[j])                                    # 5. feature열에 넣을 feature 이름 리스트 생성하기
        print('featrue name : ',iris_f[j*50])                           # feature 이름이 바뀌는 지점 값 출력해서 잘 들어가고 있는지 확인
    iris_trans['feature']=iris_f
    # st.dataframe(iris_trans)

    s1, s2= st.columns([0.5,0.5])
    s1.markdown('##### 'f'{f_list[0]}')
    s1.scatter_chart(
                data=iris_trans[iris_trans['feature']==f_list[0]].loc[:, c_list], 
                size=50
            )
    s2.markdown('##### 'f'{f_list[1]}')
    s2.scatter_chart(
                data=iris_trans[iris_trans['feature']==f_list[1]].loc[:, c_list], 
                size=50
            )
    s1.markdown('##### 'f'{f_list[2]}')
    s1.scatter_chart(
                data=iris_trans[iris_trans['feature']==f_list[2]].loc[:, c_list], 
                size=50
            )
    s2.markdown('##### 'f'{f_list[3]}')
    s2.scatter_chart(
                data=iris_trans[iris_trans['feature']==f_list[3]].loc[:, c_list], 
                size=50
            )
    # st.scatter_chart(iris_trans)

    # show Normal Distribution
    st.subheader('iris :blue[normal distribution]', divider='blue')

    # formula
    st.latex(r'''
            N(x∣μ,σ^2)≡\frac{1}{σ\sqrt{2π}}\exp[−\frac{(x−μ)^2}{2σ^2}]
            ''')

    # select chart type
    select_chart_type = st.radio(
        'select a chart type',
        ('scatter', 'line', 'hist')
    )
    norm = nc.normal_chart(iris_pd, iris_trans)


    st.markdown('### by Class')
    # select class
    select_class = st.selectbox(
        'selct a class',
        ('Setosa', 'Versicolor', 'Virginica')
    )
    norm.by_class(select_class, f_list, select_chart_type)

    st.markdown('### by Feature')
    #select Feature
    select_feature = st.selectbox(
        'select a feature',
        (f_list)
    )
    norm.by_feature(c_list, select_feature, select_chart_type)