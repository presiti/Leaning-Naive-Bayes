import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure

class normal_chart:
    def __init__(self, df, df_trans):
        self.df = df
        self.df_trans = df_trans
        self.pi = np.pi      # pie

    def by_class(self, c, f, chart):
        print('---------------------class별로 feature 정규분포 그리기')
        iris = self.df[self.df.variety==c]
        np_list = []         # 표준편차 값 list
        mu = []
        sigma = []
        
        # 정규분포 구하기
        for h in range(len(f)): # 4번 반복
            np_list.append(iris[f[h]].sort_values().to_numpy())
            mu.append(np.mean(np_list[h]))    # 평균
            sigma.append(np.std(np_list[h]))  # 표준편차

            for i in range(50): # 정규분포
                np_list[h][i] = 1/sigma[h]*np.sqrt(2*self.pi)*np.exp(-(np_list[h][i]-mu[h])**2/(2*(sigma[h]**2)))

        print('정규분포 계산 완료 :',np_list[0][0])

        # 출력하기
        st.markdown('#### 'f'{c}')

        legend = []
        fig, ax = plt.subplots()
        ax.set_title('normal distribution')

        print('그래프 종류 : ', chart)
        print()
        if(chart=='scatter'):
            chart = pd.DataFrame()
            for i in range(4):
                chart[f[i]] = np_list[i]
            st.scatter_chart(chart)

        elif(chart=='line'):
            for i in range(len(f)):
                ax.plot(np_list[i], alpha=0.7)
                legend.append(f'{iris.columns[i]}')

            ax.legend(legend)
            st.pyplot(fig)

        elif(chart=='hist'):
            for i in range(len(f)):
                ax.hist(np_list[i], alpha=0.7)
                legend.append(f'{iris.columns[i]}')
            ax.legend(legend)
            st.pyplot(fig)
        else:
            pass
        # 선택후 출력된 그래프 이미지로 다운로드 할 수 있게 하기

    def by_feature(self, c, f, chart):
        print('---------------------feature별로 feature 정규분포 그리기')
        # 하고 싶은 것
        # 선택한 feature(column)을 variety별로 나눠서 저장.=> column을 variety로, row를 feature로 변환하여 저장

        # 수행 순서
        # column이 class인 데이터 프레임 생성 
        iris = pd.DataFrame({
            i:[] for i in c
            })
        st.dataframe(iris)

        # class 명에 맞는 feature값 채워넣기 => groupbt 수행
        print(self.df.groupby('variety')) 
        
        np_list = []         # 표준편차 값 list
        mu = []
        sigma = []
        