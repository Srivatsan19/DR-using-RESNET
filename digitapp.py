from cv2 import cvtColor
import cv2
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image,ImageOps
import cv2

def load_image():
    uploaded_file = st.file_uploader(label="Select an Image")
    if uploaded_file is not None:
        image_data =Image.open(uploaded_file,)
        st.image(image_data)
        label = digit_recognizer(image_data, 'digit.h5')
        st.write(label)


def main():
    st.title('Handwritten Digit Recognition')
    load_image()

def digit_recognizer(img,weights_file):

    model = keras.models.load_model(weights_file)
    
    image=np.asarray(img,dtype=np.float32)


    resized_img=cv2.resize(image,(244,244),interpolation = cv2.INTER_AREA)

    img=resized_img/255

    final_image=np.array(img).reshape(-1,244,244,3,1)
    # run the inference
    prediction = model.predict(final_image)
    return np.argmax(prediction)



if __name__ == '__main__':
    main()

