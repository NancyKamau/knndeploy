import streamlit as st
import joblib  # or use pickle if that's what you used
import numpy as np

# Load the model (update the path if needed)
model = joblib.load('knn_model.pkl')  # replace with your actual file name

# Page title
st.title("Iris Flower Species Predictor 🌸")
st.write("Enter the flower measurements below to predict the species.")

# Input fields for the features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction:")
    st.success(f"The predicted species is: **{prediction}**")