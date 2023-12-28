import streamlit as st
import pandas as pd

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

    st.dataframe(iris_pd)
    iris_trans=pd.DataFrame()

    # 테스트1 : 깡
    iris_temp = iris_pd[iris_pd.variety==c_list[0]]             # 1. 클래스별로 우선 분리
    # st.dataframe(iris_temp)

    # 2. 분리 후 특정 행 추출 확인
    st.dataframe(iris_temp.loc[:, [f_list[0], 'variety']])          
    # print(type(iris_temp.loc[:, f_list[0]]))                # pandas Series                              
    
    # 3. 분리 후 각 feature 한 줄 넣기
    iris_trans[c_list[0]] = iris_temp.loc[:, f_list[0]]
    st.dataframe(iris_trans)

    iris_trans=pd.DataFrame()

    # 4. class별로 다 넣기
    # 그냥 넣기
    for i in range(3):
        iris_temp = iris_pd[iris_pd.variety==c_list[i]]
        iris_trans[c_list[i]] = [(iris_temp.loc[:, f_list[j]].to_numpy()) for j in range(4)]
    st.dataframe(iris_trans)

    iris_trans['feature']=f_list                              # 4. feature 행 추가하기
    st.dataframe(iris_trans)

    # 테스트1-2:concat, axis 써보기
    # st.scatter_chart(iris_trans)

    st.subheader('transpose')
    # 테스트2 : transpose()         
    iris_temp=st.dataframe(iris_pd.transpose())                 # 1. .transpos()로 행열 바꾸기

    
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