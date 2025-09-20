import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 1, 100, 25)
sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children aboard (Parch)", 0, 10, 0)
fare = st.number_input("Ticket Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

# Prepare input in the same way as training
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# One-hot encode
input_df = pd.get_dummies(input_df)

# Columns used during training
training_cols = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'Sex_female', 'Sex_male',
    'Embarked_C', 'Embarked_Q', 'Embarked_S'
]

# Align columns with training
input_df = input_df.reindex(columns=training_cols, fill_value=0)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("🎉 The passenger would have SURVIVED!")
    else:
        st.error("💀 The passenger would NOT have survived.")
