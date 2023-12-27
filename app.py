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
    print(iris_pd.isnull().sum(),'\n')


    # Dataset Info
    print('show dataset info\n')

    #show dataset head(sample)
    st.subheader('iris :violet[sample]', divider='violet')
    st.dataframe(data=iris_pd.head())

    #show dataset class info
    st.subheader('iris :violet[class info]', divider='violet')
    iris_class = iris_pd.groupby('variety')['variety'].count()
    st.dataframe(data=iris_class)

    #show dataset describe
    st.subheader('iris :violet[describe]', divider='violet')
    # print(iris_pd.describe())
    st.dataframe(iris_pd.describe())


    # Graph
    c_list = ['Setosa', 'Versicolor', 'Virginica']                          #class name list
    f_list = [iris_pd.columns[i] for i in range(4)]                         #feature name list

    #show Scatter chart
    st.subheader('iris :violet[scatter chart]', divider='violet')

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
        iris_trans[c_list[i]] = [(iris_temp.loc[:, f_list[j]]) for j in range(4)]
    st.dataframe(iris_trans)

    iris_trans['feature']=f_list[0]                              # 4. feature 행 추가하기
    st.dataframe(iris_trans.explode(c_list))

    # 테스트1-2:concat, axis 써보기
    # st.scatter_chart(iris_trans)

    
    # 테스트2 : transpose()         
    iris_temp=st.dataframe(iris_pd.transpose())                 # 1. .transpos()로 행열 바꾸기

    st.dataframe(iris_temp.grouby('variety'))

    
    # show Normal Distribution
    st.subheader('iris :violet[draw normal distribution]', divider='violet')

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

    

if __name__=='__main__':
    main()