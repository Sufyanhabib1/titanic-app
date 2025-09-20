import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details below to check survival prediction:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.slider("Ticket Fare", 0, 500, 50)

# Convert sex to numeric
sex = 0 if sex == "male" else 1

# Make prediction
if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    prediction = model.predict(input_data)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.subheader(f"Prediction: {result}")
