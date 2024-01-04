import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from scipy.stats import norm
import matplotlib as plt

from scipy.special import comb


from DatasetINFO.transposeDataset import transpose as ts

def naive_bayes(iris_pd):
    print('start naive bayes ---------------------------------')
    st.header("Naive Bayes")
    st.dataframe(iris_pd)

    # split train, test
    variety=iris_pd.variety
    data=iris_pd.drop(columns='variety')
    st.dataframe(data)

    iris_pd['category']=iris_pd['variety'].factorize()[0]
    category=iris_pd['category']
    st.dataframe(iris_pd)

    data_train, data_test, variety_train, variety_test=train_test_split(
        data, category, test_size=0.2, stratify=category, random_state=1
        )
    print('variety bincount : ',np.bincount(variety_train))
    newIris=pd.DataFrame(np.column_stack([data_train, variety_train]))

    # sort and rearrange the data based on the variety
    setosa=newIris[newIris[4]==0]
    versicolor=newIris[newIris[4]==1]
    virginica=newIris[newIris[4]==2]
    newIris=pd.concat([setosa, versicolor, virginica])

    # split data based on variety
    setosa_data=newIris[0:40]
    versicolor_data=newIris[40:80]
    virginica_data=newIris[80:120]

    # find mean
    setosa_mean=setosa_data.mean()
    versicolor_mean=versicolor_data.mean()
    virginica_mean=virginica_data.mean()

    # find standard diviation
    setosa_std=setosa_data.std()
    versicolor_std=versicolor_data.std()
    virginica_std=virginica_data.std()

    # find Likelihood
    print('finding train likelihood')
    x=[]
    likelihood=[]

    for i in range(len(newIris)):
        distribution=1
        if(i<40):                               #setosa
            mean=setosa_mean
            std=setosa_std
        if(i>=40 and i<80):                     #versicolor
            mean=versicolor_mean
            std=versicolor_std
        if(i>=80 and i < 120):                  #virginica
            mean=virginica_mean
            std=virginica_std
        
        for j in range(4):
            x = newIris.iloc[i]
            a = ((x[j]-mean[j])**2)/(2*std[j]**2)
            b = math.sqrt(2*math.pi*(std[j]**2))
            y = math.exp(-a)/b
            distribution=distribution*y
        likelihood.append(distribution)
        x=[]
    st.dataframe(likelihood)

    # find priori probability
    setosa_priori = len(setosa_data)/len(newIris)
    versicolor_priori = len(versicolor_data)/len(newIris)
    virginica_priori = len(virginica_data)/len(newIris)

    print('setosa_priori :', setosa_priori)
    print('versicolor_priori :', versicolor_priori)
    print('virginica_priori :', virginica_priori)
    print()

    # rearrange the data into groups based on the variety
    newTest = pd.DataFrame(np.column_stack([data_test, variety_test]))
    setosa=newTest[newTest[4]==0]
    versicolor=newTest[newTest[4]==1]
    virginica=newTest[newTest[4]==2]
    newTest=pd.concat([setosa, versicolor, virginica])
    
    # find likelihood for test data
    print('finding test likelihood')
    testLikelihood=[]
    x=[]
    testPosterior=[]
    posteriorVariety=[]

    for i in range(len(newTest)):
        for c in range(3):
            if(c==0):
                mean=setosa_mean
                std=setosa_std
                priori=setosa_priori
            if(c==1):
                mean=versicolor_mean
                std=versicolor_std
                priori=versicolor_priori
            if(c==2):
                mean=virginica_mean
                std=virginica_std
                priori=virginica_priori
            distribution=1

            for j in range(4):
                x=newTest.iloc[i]
                a=((x[j]-mean[j])**2)/(2*std[j]**2)
                b=math.sqrt(2*math.pi*(std[j]**2))
                y=math.exp(-a)/b
                distribution=distribution*y
            x=[]
            testLikelihood.append(distribution)
            posterior=testLikelihood[c]*priori          # Calculate poterior values
            testPosterior.append(posterior)
            maxPosterior=testPosterior.index(max(testPosterior))
        posteriorVariety.append(maxPosterior)

        testLikelihood=[]
        testPosterior=[]
    
    # check the differences
    print("variety of original test data")
    variety_test=list(map(int, newTest[4]))
    print(variety_test)

    print("variety of Maximum Posterior values")
    print(posteriorVariety)
    print()

    # Accuarcy
    similar=0
    for i in range(len(posteriorVariety)):
        if(variety_test[i]==posteriorVariety[i]):
            similar += 1
    accuarcy=similar/(i+1)*100

    print('accuarcy : ',accuarcy)

    # fig, ax = plt.subplots()
    # for i in range(len(posteriorVariety)):
    #     ax.plot(variety_test[i])
    #     ax.plot(posteriorVariety[i])
