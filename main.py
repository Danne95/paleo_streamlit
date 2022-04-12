
#terminal:
#streamlit run main.py
#streamlit run main.py --server.port=80

#git add .
#git commit -m "doco"
#git push -u

import streamlit as st
from PIL import Image
import pandas as pd


def load_image(image_file):
    img = Image.open(image_file)
    return img

st.set_page_config(page_title="Palaeography Classification", page_icon=":crystal_ball:", layout="wide")
st.title("Welcome to our paleo web application.!!!")
st.write("site under construction!")
st.write("[Learn more >](https://en.wikipedia.org/wiki/Palaeography)")
menu = ["None","Image"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Image":
    image_file = st.file_uploader("Upload Image:", type=["png","jpg","jpeg"])
    if image_file is not None:
        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
        st.write(file_details)
        # To View Uploaded Image
        st.image(load_image(image_file))