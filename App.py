import streamlit as st
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Gravitational lensing halo substructure Classifier')

st.markdown('Gravitational lensing halo classifier. The halo substructures are classified into three classes: "No substructure", "Vortex substructure" and "Sphere substructure".')


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        valid_extensions = ["jpg", "jpeg", "png"]
        file_extension = file_uploaded.name.split(".")[-1].lower()
        if file_extension in valid_extensions:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "Glensinghalo_classifier.h5"
    model = load_model(classifier_model, compile=False)
    test_image = cv2.resize(image,(150,150))
    test_image = test_image.reshape((-1,150,150,3))
    test_image = keras.applications.resnet_v2.preprocess_input(test_image)
    class_names = [
          'no',
          'sphere',
          'vort']
    predictions = model.predict(test_image)
    scores = predictions[0]
    scores = scores.numpy()    
    result = f"{class_names[np.argmax(scores)]} substructure with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result









    

if __name__ == "__main__":
    main()
