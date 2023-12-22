import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure

iris = pd.DataFrame()
np_list = []         # 표준편차 값 list
mu = []
sigma = []
pi = np.pi      # pie

def by_class(df, c, f, chart):
    iris = df[df['variety']==c]

    # 정규분포 구하기
    for h in range(len(f)): # 4번 반복
        np_list.append(iris[f[h]].sort_values().to_numpy())
        mu.append(np.mean(np_list[h]))    # 평균
        sigma.append(np.std(np_list[h]))  # 표준편차

        for i in range(50): # 정규분포
            np_list[h][i] = 1/sigma[h]*np.sqrt(2*pi)*np.exp(-(np_list[h][i]-mu[h])**2/(2*(sigma[h]**2)))

    print('정규분포 완료',np_list[0][0])

    # 출력하기
    st.markdown('#### 'f'{c}')

    legend = []
    fig, ax = plt.subplots()
    ax.set_title('normal distribution')

    print('select chat : ', chart)
    if(chart=='scatter'):
        chart = pd.DataFrame()
        for i in range(4):
            chart[f[i]] = np_list[i]
        return st.scatter_chart(chart)

    elif(chart=='line'):
        for i in range(len(np)):
            ax.plot(np_list[i], alpha=0.7)
            legend.append(f'{iris.columns[i]}')

        ax.legend(legend)
        return st.pyplot(fig)

    elif(chart=='hist'):
        for i in range(len(np)):
            ax.hist(np_list[i], alpha=0.7)
            legend.append(f'{iris.columns[i]}')

        ax.legend(legend)
        return st.pyplot(fig)
    else:
        pass
    # 선택후 출력된 그래프 이미지로 다운로드 할 수 있게 하기

def by_feature(df, c, f, chart):
    iris.append(df[df['variety']==c[i]] for i in c) 
    print(iris)