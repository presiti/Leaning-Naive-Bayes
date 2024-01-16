import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image

from DatasetINFO import datasetInfo as di
from NaiveBayes import naiveBayes as nb


fi = Image.open("image/iris.png")
def main():
    print('=============================start stremlit')
    # if 'user_menu' not in st.session_state: # 선택했던 메뉴 저장 state 생성 및 초기화
    #     st.session_state['user_menu'] = ''
    # check_reload=1
    st.set_page_config(
        page_title="Analyzing iris data",
        page_icon=fi
    )

    st.title("IRIS Dataset! :bouquet:")
    # st.image('image/iris.png', width=50)
    print('read datset --------------------------')
    iris_pd = pd.read_csv("dataset/iris.csv")
    print(iris_pd.head(),'\n')
    with st.sidebar:
        select_menu=option_menu(
            "Menu",
            ("Dataset INFO", "Naive Bayes"),        # menu name
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
    
    print('select menu :', select_menu)
    
    
    # 메뉴 선택 후 보여줄 페이지
    # print(st.session_state['user_menu'])
    if(select_menu=="Dataset INFO"):
        print()
        di.dataset_info(iris_pd) 

    elif(select_menu=="Naive Bayes"):
        print()
        nb.naive_bayes(iris_pd)

    
    
    

if __name__=='__main__':
    main()