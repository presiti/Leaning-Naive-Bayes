import streamlit as st

def naive_bayes(iris_pd):
    st.header("Naive Bayes")
    st.dataframe(iris_pd)