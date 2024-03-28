import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import time
fig = plt.figure()


with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Gravitational lensing halo substructure Classifier')

st.markdown('Gravitational lensing halo classifier. The halo substructures are classified into three classes: "No substructure", "Vortex substructure" and "Sphere substructure".')


with st.container():
    st.write('Some examples to try')
    st.image(['sample_images\\no.png','sample_images\sphere.png','sample_images\\vort.png'],caption=['no substructure','sphere substructure','vort substructure'])


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        valid_extensions = ["jpg", "jpeg", "png"]
        file_extension = file_uploaded.name.split(".")[-1].lower()
        if file_extension in valid_extensions:
            file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(150,150))
    
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
    model = load_model(classifier_model) #might want to try loading with weights only next time, [LN look into it !!! mate]
    test_image = image.reshape((-1,150,150,3))
    print(np.shape(test_image))
    class_names = [
          'no',
          'sphere',
          'vort']
    predictions = model.predict(test_image)
    print('predicitions is',predictions[0])
    scores = predictions[0]
    result = f"The above lensed image has a **{class_names[np.argmax(scores)]} substructure** with a { (100 * np.max(scores)).round(2) } % confidence in prediction." 
    return result









    

if __name__ == "__main__":
    main()
