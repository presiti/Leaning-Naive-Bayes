import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure
# from scipy.stats import norm
def main():
    print('======================= start')
    iris_pd = pd.read_csv("iris.csv")

    st.title("IRIS Dataset! :bouquet:")
    # dataset source
    st.markdown('Dataset Sources : https://gist.github.com/netj/8836201')

    # null check
    print('결측치 확인')
    print(iris_pd.isnull().sum())
    print()

    #show dataset head
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


    iris_Setosa = iris_pd[iris_pd['variety']=='Setosa']
    # iris_Versicolor = iris_pd[iris_pd['variety']=='Versicolor']
    # iris_Virginica = iris_pd[iris_pd['variety']=='Virginica']


    #show scatter chart
    st.subheader('iris :violet[scatter chart]', divider='violet')
    # st.scatter_chart(iris_pd)
    

    #show normal distribution
    st.subheader('iris :violet[draw normal distribution]', divider='violet')

    x1 = iris_Setosa['sepal.length'].sort_values().to_numpy()
    print('class : Setosa, feature : sepal.length')
    print(x1)
    print()
    # st.markdown("x1 length")
    # st.caption(len(x1))
    # print('==================== x1 type: ',type(x1))
    
    mu = np.mean(x1)        # 평균
    print('평균 : ',mu)
    sigma = (np.std(x1))    # 표준편차
    print('표준편차 : ',sigma)
    pi = np.pi              # 파이
    print()

    # 정규분포 수식
    st.latex(r'''
            N(x∣μ,σ^2)≡\frac{1}{σ\sqrt{2π}}\exp[−\frac{(x−μ)^2}{2σ^2}]
            ''')

    # 정규분포(y) 구하기
    y1 = x1
    for i in range(len(x1)):
        y1[i] = 1/sigma*np.sqrt(2*pi)*np.exp(-(y1[i]-mu)**2/(2*(sigma**2)))
    print('정규분포')
    print(y1)
    print()

    #출력하기
    st.markdown("##### matplot")
    st.markdown("Setosa-sepal.length Origin")
    fig, ax = plt.subplots()
    ax.plot(x1)
    st.pyplot(fig)

    st.markdown("Setosa-sepal.normal distribution")
    fig, ax = plt.subplots()
    ax.plot(y1)
    st.pyplot(fig)

    st.scatter_chart(y1)
    
if __name__=='__main__':
    main()