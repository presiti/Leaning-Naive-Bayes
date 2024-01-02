import streamlit as st
import pandas as pd
import numpy as np

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

    # st.dataframe(iris_pd)
    iris_np=np.empty(200)
    iris_trans=pd.DataFrame()

    # 데이터셋 변환_1
    # 1. 클래스 하나 분리해서 돌리기
    iris_temp = iris_pd[iris_pd.variety==c_list[0]].transpose()             
    st.dataframe(iris_temp)

    # print(type(iris_temp.loc[:, f_list[0]]))                # pandas Series
    # 2. feture 분리 해서 한 줄로 붙이기
    for i in range(4):
        f_np=iris_temp.iloc[i]                                          # feature 분리
        print(f_list[i],'first value : ', f_np[0])
        for j in range(50):
            iris_np[j+i*50] = f_np[j]                                   # 값을 한개씩 빼서 4*50개를 한 줄로 붙이기
            # 문제점 1. 잘 만들어지는데 소수점 뒤로 숫자가 없으면 1.0이 1로 들어감.
            # 해결방법 : type 출력해보기
    
    # 3. data farme으로 만들어 주기
    iris_trans[c_list[0]] = iris_np
    st.dataframe(iris_trans)

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
    norm = nc.normal_chart(iris_pd)


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
    # norm.by_feature(c_list, select_feature, select_chart_type)