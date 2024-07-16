import streamlit as st

def main():
    st.title("Mask Classifier")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Perform prediction on the uploaded image
        # Replace the code below with your prediction logic
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Making predictions...")
        # Add your prediction logic here
        
        # Display the prediction results
        st.write("Prediction: Mask")
    
    st.button("Make Predictions")

if __name__ == "__main__":
    main()