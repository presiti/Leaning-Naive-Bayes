import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from decimal import Decimal, ROUND_HALF_UP

import DatasetINFO.normalChart as nc
import DatasetINFO.transposeDataset as td

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

    

    # 데이터셋 변환
    iris_trans=td.transpose(iris_pd, c_list, f_list)
    st.dataframe(iris_trans)
    
    scatter_1 = st.toggle('show sactter version 1')
    if scatter_1:
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
    
    st.markdown('### sepal')
    setosa = iris_pd[iris_pd['variety']=='Setosa']
    versicolor = iris_pd[iris_pd['variety']=='Versicolor']
    virginica = iris_pd[iris_pd['variety']=='Virginica']
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.scatter(setosa['sepal.width'],setosa['sepal.length'])
    ax1.scatter(versicolor['sepal.width'],versicolor['sepal.length'])
    ax1.scatter(virginica['sepal.width'],virginica['sepal.length'])
    # st.pyplot(fig)

    st.markdown('### petal')
    # fig, ax = plt.subplots()
    ax2.scatter(setosa['petal.width'],setosa['petal.length'])
    ax2.scatter(versicolor['petal.width'],versicolor['petal.length'])
    ax2.scatter(virginica['petal.width'],virginica['petal.length'])
    st.pyplot(fig)

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
    np_data = norm.by_class(select_class, f_list)
    norm.show_chart(
        'class', 
        np_data, 
        select_class, 
        f_list, 
        select_chart_type
        )

    st.markdown('### by Feature')   
    #select Feature
    select_feature = st.selectbox(
        'select a feature',
        (f_list)
    )
    norm.by_feature(c_list, select_feature, select_chart_type)