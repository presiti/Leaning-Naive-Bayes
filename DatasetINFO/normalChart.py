import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from bokeh.plotting import figure

class normal_chart:
    def __init__(self, df, df_trans):
        self.df = df
        self.df_trans = df_trans
        self.pi = np.pi      # pie

    def by_class(self, c, f):
        print('--------------------- class별로 feature 정규분포 그리기')
        iris = self.df[self.df.variety==c]                          # 1. 선택된 class 데이터만 분리하기
        np_list = []                                                # 표준편차 값 리스트
        mu = []
        sigma = []
        
        # 정규분포 구하기
        for h in range(len(f)): # 0~3, feature 수만큼 반복
            np_list.append(iris[f[h]].sort_values().to_numpy())     # 2. feature별 데이터 분리, 오름차순 정렬, 정규분포 계산을 위한 numpy 변환
            mu.append(np.mean(np_list[h]))                          # 3. feature별 평균
            sigma.append(np.std(np_list[h]))                        # 4. feature별 표준편차

            for i in range(50): # 정규분포
                np_list[h][i] = 1/sigma[h]*np.sqrt(2*self.pi)*np.exp(-(np_list[h][i]-mu[h])**2/(2*(sigma[h]**2)))

        print('정규분포 계산 완료 / sample :',np_list[0][0])

        # 출력하기
        st.markdown('#### 'f'{c}')

        # legend = []
        # fig, ax = plt.subplots()
        # ax.set_title('normal distribution')

        # print('그래프 종류 : ', chart)
        # print()
        # if(chart=='scatter'):
        #     chart = pd.DataFrame()
        #     for i in range(4):
        #         chart[f[i]] = np_list[i]
        #     st.scatter_chart(chart)

        # elif(chart=='line'):
        #     for i in range(4):
        #         ax.plot(np_list[i], alpha=0.7)
        #         legend.append(f[i])

        #     ax.legend(legend)
        #     st.pyplot(fig)

        # elif(chart=='hist'):
        #     for i in range(4):
        #         ax.hist(np_list[i], alpha=0.7)
        #         legend.append(f'{iris.columns[i]}')
        #     ax.legend(legend)
        #     st.pyplot(fig)
        # else:
        #     pass
        return np_list

    def by_feature(self, c, f, chart):
        print('--------------------- feature별로 feature 정규분포 그리기')
        iris = self.df_trans[self.df_trans.feature==f]              # 1. 선택된 feature 데이터 분리
        np_list = []                                                # 표준편차 값 list
        mu = []
        sigma = []

        # 정규분포 구하기
        for i in range(len(c)): # 0~2, class 수만큼 반복
            np_list.append(iris[c[i]].sort_values().to_numpy())     # 2. class별 데이터 분리, 오름차순 정렬, 정규분포 계산을 위한 numpy 변환
            mu.append(np.mean(np_list[i]))                          # 3. class별 평균
            sigma.append(np.std(np_list[i]))                        # 4. class별 표준편차
            for j in range(50):
                np_list[i][j] = 1/sigma[i]*np.sqrt(2*self.pi)*np.exp(-(np_list[i][j]-mu[i])**2/(2*(sigma[i]**2)))

        print('정규분포 계산 완료 / sample :', np_list[0][0])

        # 출력하기
        st.markdown('#### 'f'{f}')

        legend = []
        fig, ax = plt.subplots()
        ax.set_title('normal distribution')

        print('그래프 종류 : ', chart)
        print()
        if(chart=='scatter'):
            chart = pd.DataFrame()
            for i in range(3):
                chart[c[i]] = np_list[i]
            st.scatter_chart(chart)

        elif(chart=='line'):
            for i in range(3):
                ax.plot(np_list[i], alpha=0.7)
                legend.append(f'{iris.columns[i]}')

            ax.legend(legend)
            st.pyplot(fig)

        elif(chart=='hist'):
            for i in range(3):
                ax.hist(np_list[i], alpha=0.7)
                legend.append(f'{iris.columns[i]}')
            ax.legend(legend)
            st.pyplot(fig)
        else:
            pass
    
    def show_chart(self, data_by, norm, c, f, chart):
        print('chart start----------------')
        print(self, '\n')
        len=0
        x_name=[]
        legend = []
        fig, ax = plt.subplots()
        ax.set_title('normal distribution')
        print('그래프 종류 : ', chart,'\n')
        if(data_by=='class'): 
            len=len(f)
            x_name=f
        if(data_by=='feature'): 
            len=len(c)
            x_name=c
        
        if(chart=='scatter'):
            chart = pd.DataFrame()
            for i in range(4):
                chart[f[i]] = norm[i]
            st.scatter_chart(chart)

        elif(chart=='line'):
            for i in range(4):
                ax.plot(norm[i], alpha=0.7)
                legend.append(x_name[i])

            ax.legend(legend)
            st.pyplot(fig)

        elif(chart=='hist'):
            for i in range(4):
                ax.hist(norm[i], alpha=0.7)
                legend.append(x_name[i])
            ax.legend(legend)
            st.pyplot(fig)
        else:
            pass