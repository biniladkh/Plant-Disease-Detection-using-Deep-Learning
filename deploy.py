import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

model=load_model(r'D:\Plant Disease Prediction\model.h5')
file_path = r'C:\Users\bnlad\Downloads\plant pathology dataset\images\Train_484.jpg'
path =  r'C:\Users\bnlad\Downloads\plant pathology dataset\images'
def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    categories=['Healthy','Multiple Disease','Rust','Scab']
    pred_class = result.argmax()
    output=categories[pred_class]
    return output

def main():
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.title("Plant Disease Prediction")
    st.write("**This app predicts the following categories Healthy, Multiple Disease, Rust and  Scab**")
    image_file = st.file_uploader("Upload image to predict", type=['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:
        
        if st.button("Predict"):
            st.image(image_file, use_column_width = True)
            # st.write(image_file.name)
            result = model_predict(os.path.join(path,image_file.name),model)
            st.success("The prediction is {}".format(result))

if __name__ == "__main__":
    main()