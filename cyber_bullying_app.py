import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import xgboost

page = st.sidebar.selectbox(
    'Select a page:',
    ('Home', 'Detection App', 'About Us')
)
if page == 'About Us':
    st.title("Meet the Team")
    nati = Image.open('st-images/nati.jpg')
    rebecca = Image.open('st-images/rebecca.jpg')
    julia = Image.open('st-images/julia.png')
    christian = Image.open('st-images/christian.jpg')
    col1, col2, col3, col4 = st.beta_columns([1,1,1,1])
    #st.image([nati,rebecca,julia,christian])
    col1.image(nati,width=162)
    col2.image(rebecca,use_column_width=True)
    col3.image(julia,use_column_width=True)
    col4.image(christian,use_column_width=True)
    st.markdown('''
    We are data scientists currently enrolled in General Assembly's Data Science Immersive course.
    Throughout this 12-week program, we have learned everything from simple regression to Neural Neworks.
    We hope this project will aid in the fight against cyberbullying.

    Have questions about the project? Email a team member here:
    [Nati](#mailto:nati.marcus24@gmail.com), [Rebecca](#mailto:bb@gmail.com), [Julia](#mailto:djdjdj@dndn.com), [Christian](#mailto:chris@mail.com)
    ''')
    ga_logo = Image.open('st-images/ga-logo.png').convert('RGBA')
    st.image(ga_logo,width=700,caption='source: https://www.vhv.rs/viewpic/hxihoiJ_vector-point-assembly-general-assembly-logo-png-transparent/')

if page == 'Home':
    st.title('Detecting Cyberbullying')

    image = Image.open('st-images/Cyberbullying_02.png')
    st.image(image, caption='source: https://www.avast.com/c-cyberbullying')

    st.markdown('''
    ## **Problem Statement**

    Many content providers on the internet give users the ability to write comments with the goal of promoting healthy discussion. Unfortunately, comment sections are prone to abuse, and many users experience cyberbullying in the form of toxic, aggressive, or attack-laden comments. Programmatic detection of cyberbullying has a strong use case for content providers who seek to remove harmful content from their comment sections. In this project, we address this problem by developing a set of models to predict the likelihood that a comment contains a personal attack, toxic language, or aggressive tone.
    ''')

    st.markdown('''
    ## **Methodology**
    This study builds on Wikimedia's Detox Project, where researchers utilized comments from Wikipedia Talk to analyze whether the contents of the comments were considered an attack or not an attack.
    Using data collected by Wikimedia, we were able to construct three separate models on comments that were perceived to be an attack, aggressive, or toxic.
    ''')

    st.markdown("## **Data & Code**")
    st.markdown("All data and code can be found on our [github repo](https://github.com/ctnormand1/cyber_bullying_detection)")

if page == 'Detection App':
    st.title("Detection App")
    st.write('''This page uses machine learning to determine how likely a piece of text is to be either aggressive, toxic, or an attack.''')

    selectbox = st.selectbox('Model',['Attack Model','Toxicity Model','Aggression Model'])

    if selectbox == "Attack Model":
        #opening pickle file for attack model
        with open('model_for_app/xgboost_attacks.pkl', mode='rb') as pickle_in:
            pipe = pickle.load(pickle_in)
        text = st.text_input("Is this comment an attack?")
        attack_result = pipe.predict([text])[0]
        if attack_result == 0:
            response = 'not an attack'
        else:
            response = 'AN ATTACK'

        st.write(f'This comment is {response}')
        if response == 'AN ATTACK':
            #inserting attack image
            image = Image.open('st-images/attack.png')
            col1, col2, col3 = st.beta_columns([1,1,1])
            col2.image(image, caption='source: https://emoji.gg/emoji/PikaAttack')

    elif selectbox == "Toxicity Model":
        #opening pickle file for toxicity model
        with open('model_for_app/xgboost_toxic.pkl', mode='rb') as pickle_in:
            pipe = pickle.load(pickle_in)
        text = st.text_input("Is this comment toxic?")
        #saving the prediction from toxicity model
        toxic_result = pipe.predict([text])[0]
        if toxic_result == 0:
            response = 'not toxic'
        else:
            response = 'TOXIC'

        st.write(f'This comment is {response}')
        if response == "TOXIC":
            image = Image.open('st-images/toxicity.png')
            col1, col2, col3 = st.beta_columns([1,1,1])
            col2.image(image,caption='source: https://www.emojipng.com/preview/990569')

    elif selectbox == "Aggression Model":
        # opening pickle file for aggression model
        with open('model_for_app/xgboost_aggression.pkl', mode='rb') as pickle_in:
            pipe = pickle.load(pickle_in)

        text = st.text_input("Is this comment aggressive?")
        #saving the prediction from the aggression model
        aggressive_result = pipe.predict([text])[0]
        if aggressive_result == 0:
            response = 'not aggressive'
        else:
            response = 'AGGRESSIVE'

        st.write(f'This comment is {response}')
        if response == "AGGRESSIVE":
            image = Image.open('st-images/aggression.jpeg')
            col1, col2, col3 = st.beta_columns([1,1,1])
            col2.image(image,use_column_width=True,caption='source: https://www.seekpng.com/ipng/u2q8q8e6o0w7u2e6_aggression-clip-art-angry-and-bite-the-smiley/')
