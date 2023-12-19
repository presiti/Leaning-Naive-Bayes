import streamlit as st

import numpy as np
import pandas as pd

def main():
    iris_pd = pd.read_csv("iris.csv")

    st.title("IRIS Dataset! :bouquet:")
    # dataset source
    st.markdown('Dataset Sources : https://gist.github.com/netj/8836201')

    # null check
    print(iris_pd.isnull().sum())

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
    iris_Versicolor = iris_pd[iris_pd['variety']=='Versicolor']
    iris_Virginica = iris_pd[iris_pd['variety']=='Virginica']

    #show scatter chart
    st.subheader('iris :violet[scatter chart]', divider='violet')
    # st.scatter_chart(iris_pd)
    
    #show normal distribution
    st.subheader('iris :violet[draw normal distribution]', divider='violet')

    x1 = iris_Setosa['sepal.length']
    st.dataframe(x1)
    st.markdown("x1 length")
    st.caption(len(x1))
    
    mean = np.mean(x1)
    st.markdown(mean)

    var=(np.var(x1))
    st.markdown(var)
    
    st.latex(r'''
            N(x∣μ,σ^2)≡\frac{1}{σ\sqrt{2π}}\exp[−\frac{(x−μ)^2}{2σ^2}]
            ''')

    norm = x1
    for i in range(len(x1)):
        norm[i] = (1/var*np.sqrt(2*np.pi))*np.exp(-((x1[i]-mean)**2/2*var))
    st.dataframe(norm)
    st.line_chart(norm)
    
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    st.dataframe(chart_data)
    st.line_chart(chart_data)
if __name__=='__main__':
    main()