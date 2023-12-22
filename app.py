import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure
# from scipy.stats import norm

def normal_chart(df, c, f, chart):
    iris = df[df['variety']==c]
    # iris_Versicolor = iris_pd[iris_pd['variety']=='Versicolor']
    # iris_Virginica = iris_pd[iris_pd['variety']=='Virginica']

    f_np = []       # feature별 표준편차 값 list
    mu = []
    sigma = []

    pi = np.pi      # pie
    length = 50

    # 정규분포 구하기
    for h in range(len(f)): # 4번 반복
        f_np.append(iris[f[h]].sort_values().to_numpy())
        mu.append(np.mean(f_np[h]))    # 평균
        sigma.append(np.std(f_np[h]))  # 표준편차
        
        for i in range(length): # 정규분포
            f_np[h][i] = 1/sigma[h]*np.sqrt(2*pi)*np.exp(-(f_np[h][i]-mu[h])**2/(2*(sigma[h]**2)))

    # 출력하기
    st.markdown('#### 'f'{c}')

    legend = []
    fig, ax = plt.subplots()
    ax.set_title('normal ')

    print('select chat : ', chart)
    if(chart=='scatter'):
        chart = pd.DataFrame()
        for i in range(4):
            chart[f[i]] = f_np[i]
        st.scatter_chart(chart)
        
    elif(chart=='line'):
        for i in range(len(f_np)):
            ax.plot(f_np[i], alpha=0.7)
            legend.append(f'{iris.columns[i]}')

        ax.legend(legend)
        st.pyplot(fig)

    elif(chart=='hist'):
        for i in range(len(f_np)):
            ax.hist(f_np[i], alpha=0.7)
            legend.append(f'{iris.columns[i]}')

        ax.legend(legend)
        st.pyplot(fig)
    else:
        pass
    
    # 둘중 선택할 수 있게 하기 - selectbox
    # 선택후 출력된 그래프 이미지로 다운로드 할 수 있게 하기

def main():
    print('=============================start stremlit')
    iris_pd = pd.read_csv("iris.csv")

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
    print('===================출력')
    print(iris_class)
    print()

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
    
    # select class
    select_class = st.selectbox(
        'selct a class',
        ('Setosa', 'Versicolor', 'Virginica')
    )

    # select chart type
    select_normal_chart = st.radio(
        'select a chart type',
        ('scatter','line','hist')
    )
    # c_list = ['Setosa', 'Versicolor', 'Virginica']
    f_list = ['sepal.length', 'sepal.width','petal.length','petal.width']
    normal_chart(iris_pd, select_class, f_list, select_normal_chart)
    

if __name__=='__main__':
    main()