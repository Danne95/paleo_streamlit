#pipreqs ./ --force #for requirements
"""
    keras<=2.8.0
    numpy<=1.17.4
    opencv-python-headless<=4.5.4.60
    pandas<=1.3.5
    Pillow<=9.1.0
    streamlit<=1.8.1
    tensorflow<=2.8.0
"""

# bash commands for streamlit-server
#terminal:
#streamlit run main2.py
#streamlit run paleo_site/main2.py
#nohup streamlit run yourscript.py
#streamlit run paleo_site/main2.py --server.port=80

# git commands for streamlit-cloud
#cd paleo_site
#git pull
#git add .
#git commit -m "text"
#git push -u



import streamlit as st
from PIL import Image
#import pandas as pd
#import tensorflow as tf
#import keras
#from myConfig.predict import predict_image, revert
import myConfig.predict as myPred
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def start_prediction(image_file, image_name):
    image_file = cv2.cvtColor(np.array(image_file), cv2.COLOR_RGB2BGR)
    with rows[3][0]:
        st.header("Prediction:")
        prediction = myPred.predict_image(image_file,myPred.class_model,myPred.subclass_model)
        st.write("We think \'{}\' is: \n".format(image_name)+myPred.revert((prediction["class"],prediction["subclass"])))
        st.write("Class confidence: {:.2f}\n".format(max(prediction["class_bins"])*100))
        st.write("Sublass confidence: {:.2f}\n".format(max(prediction["subclass_bins"])*100))

def on_model_select():
    with rows[2][0]:
        st.subheader("Step 3:click the predict button to the left")
    with rows[2][1]:
        st.button(label="Predict", on_click=start_prediction, args = (img,image_file.name))

st.set_page_config(page_title="Palaeography Classification", page_icon=":crystal_ball:", layout="wide") #has to be first line!

st.title("Hebrew paleography classifier web application!")
st.write("[Learn more about paleography>](https://en.wikipedia.org/wiki/Palaeography)")

#scrolldown_img = Image.open("scrolldown.gif")

rows = [st.columns([3,4]),st.columns([3,4]),st.columns([3,4]),st.columns([3,4])]
with rows[0][0]:
    st.header("Instructions:")
    st.subheader("Step 1:upload a document image")


with rows[0][1]:
    st.header("Process:")
    image_file = st.file_uploader("Upload Image:", type=["png","jpg","jpeg"])
    if image_file is not None:
        with rows[0][0]:
            st.image("scrolldown.gif")
        # process image file
        img = Image.open(image_file)
        # resize image
        #resized = cv2.resize(img, (int(img.shape[1] / 20), int(img.shape[0] / 20)), interpolation = cv2.INTER_AREA)
            
        # To view uploaded image
            #st.image(resized, channels='BGR', use_column_width = True)
        st.image(img, use_column_width = True)
        with rows[1][1]:
            global choice
            choice = st.radio("Model selection:", (' ', 'SCE', 'BGU', 'Both'),on_change =on_model_select)
        with rows[1][0]:
            st.subheader("Step 2:choose a model from the radio buttons")