import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
# import numpy as np

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the model
# model = load_model("keras_Model.h5", compile=False)

# # Load the labels
# class_names = open("labels.txt", "r").readlines()

# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# # Replace this with the path to your image
# image = Image.open("testImages/393-with-mask.jpg").convert("RGB")

# # resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# # turn the image into a numpy array
# image_array = np.asarray(image)

# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# # Load the image into the array
# data[0] = normalized_image_array

# # Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# # Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)

 #convert this into a streamlit app
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("MaskClassifier.labels.txt", "r").readlines()

def predict(image):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.write("Class:", class_name[2:])
    st.write("Confidence Score:", confidence_score)

if uploaded_file is not None:
    if st.button("Submit"):
        classify_image(uploaded_file)

else:
    uploaded_file = os.listdir("uploads")
    if len(uploaded_files) > 0:
        selected_file = st.sidebar.selectbox("Select an image", uploaded_files)
        image = Image.open("uploads/" + selected_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.write("Please upload an image to classify")

    classify_image("uploads/" + selected_file)