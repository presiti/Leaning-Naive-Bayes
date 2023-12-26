import normalChart as nc

import streamlit as st
import pandas as pd


def main():
    print('=============================start stremlit')
    iris_pd = pd.read_csv("dataset/iris.csv")

    st.title("IRIS Dataset! :bouquet:")
    # dataset source
    st.markdown('Dataset Sources : https://gist.github.com/netj/8836201')

    # null check
    print('결측치 확인')
    print(iris_pd.isnull().sum())
    print()


    # Dataset Info

    #show dataset head
    st.subheader('iris :violet[sample]', divider='violet')
    st.dataframe(data=iris_pd.head())

    #show dataset class info
    st.subheader('iris :violet[class info]', divider='violet')
    iris_class = iris_pd.groupby('variety')['variety'].count()
    st.dataframe(data=iris_class)
    st.dataframe(data=iris_pd.groupby('variety'))

    #show dataset describe
    st.subheader('iris :violet[describe]', divider='violet')
    # print(iris_pd.describe())
    st.dataframe(iris_pd.describe())


    # Graph

    #show scatter chart
    st.subheader('iris :violet[scatter chart]', divider='violet')
    # st.scatter_chart(iris_pd)

    # show normal distribution
    st.subheader('iris :violet[draw normal distribution]', divider='violet')

    # formula
    st.latex(r'''
            N(x∣μ,σ^2)≡\frac{1}{σ\sqrt{2π}}\exp[−\frac{(x−μ)^2}{2σ^2}]
            ''')
    c_list = ['Setosa', 'Versicolor', 'Virginica']                          #class name list
    f_list = [iris_pd.columns[i] for i in range(4)]                         #feature name list
    print('===================출력')
    print(f_list)
    print()

    # select chart type
    select_chart_type = st.radio(
        'select a chart type',
        ('scatter', 'line', 'hist')
    )

    st.markdown('### by Class')
    # select class
    select_class = st.selectbox(
        'selct a class',
        ('Setosa', 'Versicolor', 'Virginica')
    )
    nc.by_class(iris_pd, select_class, f_list, select_chart_type)
    
    

    st.markdown('### by Feature')
    #select Feature
    select_feature = st.selectbox(
        'select a feature',
        (f_list)
    )
    nc.by_feature(iris_pd, c_list, select_feature, select_chart_type)

    

if __name__=='__main__':
    main()