from flask import Flask, render_template, request
from segmentation import _Labels
import streamlit as st
from PIL import Image


def main():
    st.title("Advanced Image Segmentaion and Classifier Model")

    uploaded_image = st.file_uploader("Upload an image")

    if uploaded_image is not None: 
        image = Image.open(uploaded_image)
        output_string,counts = _Labels(image)

        st.write("Output:")
        st.write(output_string)
        st.write(counts)
    else:
        st.write("Please upload an image.")

if __name__ == "__main__":
    main()