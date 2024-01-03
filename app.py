import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

import datasetInfo as di
import naiveBayes as nb
from PIL import Image

fi = Image.open("image/iris.png")
ml = Image.open("image/classification.png")
def main():
    print('=============================start stremlit')
    st.set_page_config(
        page_title="Analyzing iris data",
        page_icon=fi
    )

    st.title("IRIS Dataset! :bouquet:")
    # st.image('image/iris.png', width=50)
    iris_pd = pd.read_csv("dataset/iris.csv")

    with st.sidebar:
        select_menu=option_menu(
            "Menu",
            ("dataset INFO", "Naive Bayes"),        # menu name
            icons=["bookmark", "diagram-2"],    
            menu_icon="list", 
            default_index=0,
            styles={
                "container" : {"padding":"4!important", "background-color":"#fafafa"},
                "icon" : {"color":"black", "font-size":"20px"},
                "nav-link":{"font-size":"16px", "text-align":"left", "margin":"0px","--hover-color":"#f6f2fc"},
                "nav-link-selected":{"color":"black","background-color":"#ece1fc "}
            }
        )

    # linked menu
    if(select_menu=="dataset INFO"):
        di.dataset_info(iris_pd)                    
    elif(select_menu=="Naive Bayes"):
        nb.naive_bayes(iris_pd)

    

if __name__=='__main__':
    main()