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
#streamlit run main.py
#streamlit run paleo_site/main.py
#nohup streamlit run yourscript.py
#streamlit run paleo_site/main.py --server.port=80

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

st.set_page_config(page_title="Palaeography Classification", page_icon=":crystal_ball:", layout="wide") #has to be first line!

def load_image(image_file):
    Image.open(image_file).save("img.jpg")
    # OpenCv Read
    return cv2.imread("img.jpg")

def start_prediction(image_file, image_name, col):
    with col:
        st.header("Our prediction:")
        prediction = myPred.predict_image(image_file,myPred.class_model,myPred.subclass_model)
        #print(prediction["class_bins"])
        st.write("We think \'{}\' is: \n".format(image_name)+myPred.revert((prediction["class"],prediction["subclass"])))
        st.write("Class confidence: {:.2f}\n".format(max(prediction["class_bins"])*100))
        st.write("Sublass confidence: {:.2f}\n".format(max(prediction["subclass_bins"])*100))


st.title("Welcome to our paleo project web application!")
st.write("[Learn more >](https://en.wikipedia.org/wiki/Palaeography)")
menu = ["None","model by SCE","model by BGU"]
choice = st.sidebar.selectbox("Navigator",menu)
if choice == "model by SCE":
    image_file = st.file_uploader("Upload Image:", type=["png","jpg","jpeg"])
    if image_file is not None:
        cols = st.columns(2)
        # process image file
        img = load_image(image_file)
        
        # resize image
        #resized = cv2.resize(img, (int(img.shape[1] / 20), int(img.shape[0] / 20)), interpolation = cv2.INTER_AREA)
        # To view uploaded image
        with cols[0]:
            st.header("Your manuscript:")
            #st.image(resized, channels='BGR', use_column_width = True)
            st.image(img, channels='BGR', use_column_width = True)
            st.button(label="Predict", on_click=start_prediction, args = (img,image_file.name, cols[1]))

if choice == "model by BGU":
    st.write("work in progress")