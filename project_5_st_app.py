import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

page = st.sidebar.selectbox(
    'Select a page:',
    ('Home', 'Detection App')
)
if page == 'Home':
    st.title('Detecting Cyberbullying')

    image = Image.open('Cyberbullying_02.png')
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

    def sample_output(text,options):
        return np.random.choice(options)

    selectbox = st.selectbox('Model',['Attack Model','Toxicity Model','Aggression Model'])
    if selectbox == "Attack Model":
        text = st.text_input("Is this comment an attack?")
        options = ['an attack','not attack']
        st.write(f'This comment is {sample_output(text,options)}')
        if sample_output(text,options) == "an attack":
            image = Image.open('attack.png')
            st.image(image, caption='source: https://emoji.gg/emoji/PikaAttack')
    elif selectbox == "Toxicity Model":
        text = st.text_input("Is this comment toxic?")
        options = ['toxic','not toxic']
        st.write(f'This comment is {sample_output(text,options)}')
        if sample_output(text,options) == "toxic":
            image = Image.open('toxicity.png')
            st.image(image, caption='source: https://www.emojipng.com/preview/990569')
    elif selectbox == "Aggression Model":
        text = st.text_input("Is this comment aggressive?")
        options = ['aggressive','not aggressive']
        st.write(f'This comment is {sample_output(text,options)}')
        if sample_output(text,options) == "aggressive":
            image = Image.open('aggression.jpeg')
            st.image(image, caption='source: https://www.seekpng.com/ipng/u2q8q8e6o0w7u2e6_aggression-clip-art-angry-and-bite-the-smiley/')
    # with open('models/author_pipe.pkl', mode='rb') as pickle_in:
    #     pipe = pickle.load(pickle_in)
    #
    # user_text = st.text_input('Please input some text:',
    #     value='quoth the raven...nevermore')
    #
    # predicted_author = pipe.predict([user_text])[0]
    #
    # st.write(f'You write like: {predicted_author}')
