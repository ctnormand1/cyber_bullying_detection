import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle

page = st.sidebar.selectbox(
    'Select a page:',
    ('Home', 'Detection App')
)
if page == 'Home':
    st.title('Detecting Cyberbullying')

    image = Image.open('st-images/Cyberbullying_02.png')
    #source: https://www.avast.com/c-cyberbullying
    st.image(image, caption='source: https://www.avast.com/c-cyberbullying')

    titles = ['attack','toxicity','aggression']
    high_vals = [1.0,2,2]
    low_vals = [0.0,-2.0,-2.0]

    dataframe = pd.DataFrame([high_vals,low_vals],
    columns=titles,index=['high','low'])

    st.write('''Cyberbullying .... with this app, we aim to aid companies in their ability to detect and prevent cyberbullying on social media.''')

    st.markdown("## **Methodology**")
    st.table(dataframe)

    st.markdown("## **About Us**")
    st.markdown("We are data scientists at general assembly. you can contact us [here](mailto:nati.marcus24@gmail.com). all data and code can be found at our [github repo](https://github.com/ctnormand1/cyber_bullying_detection)")

if page == 'Detection App':
    st.title("Detection App")
    st.write('''This page uses machine learning to determine how likely a piece of text is to be either aggressive, toxic, or an attack.''')

    selectbox = st.selectbox('Model',['Attack Model','Toxicity Model','Aggression Model'])
    if selectbox == "Attack Model":
        # Add in XGboost model for attacks
        with open('model_for_app/xgboost_attacks.pkl', mode='rb') as pickle_in:
            pipe = pickle.load(pickle_in)

        text = st.text_input("Is this comment an attack?")

        attack_result = pipe.predict([text])[0]
        if attack_result == 0:
            response = 'not an attack'
        else:
            response = 'AN ATTACK'

        st.write(f'This comment is {response}')
        if response == "AN ATTACK":
            image = Image.open('st-images/attack.png')
            st.image(image, caption='source: https://emoji.gg/emoji/PikaAttack')

    elif selectbox == "Toxicity Model":
        # Add in XGboost model for toxicity
        with open('model_for_app/xgboost_toxic.pkl', mode='rb') as pickle_in:
            pipe = pickle.load(pickle_in)

        text = st.text_input("Is this comment toxic?")
        toxic_result = pipe.predict([text])[0]
        if toxic_result == 0:
            response = 'not toxic'
        else:
            response = 'TOXIC'

        st.write(f'This comment is {response}')
        if response == "TOXIC":
            image = Image.open('st-images/toxicity.png')
            st.image(image, caption='source: https://www.emojipng.com/preview/990569')

    elif selectbox == "Aggression Model":
        # Add in XGboost model for aggression
        with open('model_for_app/xgboost_aggression.pkl', mode='rb') as pickle_in:
            pipe = pickle.load(pickle_in)

        text = st.text_input("Is this comment aggressive?")

        aggressive_result = pipe.predict([text])[0]
        if aggressive_result == 0:
            response = 'not aggressive'
        else:
            response = 'AGGRESSIVE'

        st.write(f'This comment is {response}')
        if response == "AGGRESSIVE":
            image = Image.open('st-images/aggression.jpeg')
            st.image(image, caption='source: https://www.seekpng.com/ipng/u2q8q8e6o0w7u2e6_aggression-clip-art-angry-and-bite-the-smiley/')
