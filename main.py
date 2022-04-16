
#terminal:
#streamlit run main.py
#streamlit run main.py --server.port=80

#pipreqs ./ --force #for requirements
"""
    keras<=2.8.0
    numpy<=1.17.4
    opencv_python<=4.5.4.60
    pandas<=1.3.5
    Pillow<=9.1.0
    streamlit<=1.8.1
    tensorflow<=2.8.0
    opencv-python<=4.5.4
"""

#git pull
#git add .
#git commit -m "text"
#git push -u


import streamlit as st
from PIL import Image
import pandas as pd
import tensorflow as tf
import keras
from myConfig.predict import predict_image

class_model = 1 #keras.models.load_model("/home/historicalmanuscripts/paleo_site/keras_models_v1/Priya_Dwivedi_classes_v1_10e_05d")
subclass_model = 2 #keras.models.load_model("/home/historicalmanuscripts/paleo_site/keras_models_v1/Priya_Dwivedi_subclasses_v1_10e_05d")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def start_prediction(image_file):
    result = predict_image(image_file,1,2)
    return result

st.set_page_config(page_title="Palaeography Classification", page_icon=":crystal_ball:", layout="wide")
st.title("Welcome to our paleo project web application!!!")
st.write("site under construction!!")
st.write("[Learn more >](https://en.wikipedia.org/wiki/Palaeography)")
menu = ["None","Image"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Image":
    image_file = st.file_uploader("Upload Image:", type=["png","jpg","jpeg"])
    if image_file is not None:
        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
        st.write(file_details)
        # To view uploaded image
        st.image(load_image(image_file))
        # To view prediction
        st.write(start_prediction(image_file))