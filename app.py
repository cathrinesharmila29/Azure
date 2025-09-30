import streamlit as st
import pickle
import pandas as pd

with open("heart_disease_models.pkl", "rb") as f:
    saved_objects = pickle.load(f)

rf_model = saved_objects["rf_model"]
logreg_model = saved_objects["log_reg_model"]
scaler = saved_objects["scaler"]
pca = saved_objects["pca"]

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# Input fields
male = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.number_input("Age", 20, 100, 30)
education = st.selectbox("Education Level", [1, 2, 3, 4])
currentSmoker = st.selectbox("Current Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
cigsPerDay = st.number_input("Cigarettes per Day", 0, 60, 0)
BPMeds = st.selectbox("On BP Medication", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
prevalentStroke = st.selectbox("History of Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
prevalentHyp = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
totChol = st.number_input("Total Cholesterol", 100.0, 700.0, 200.0)
sysBP = st.number_input("Systolic BP", 80.0, 250.0, 120.0)
diaBP = st.number_input("Diastolic BP", 50.0, 150.0, 80.0)
BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
heartRate = st.number_input("Heart Rate", 40, 200, 70)
glucose = st.number_input("Glucose", 50.0, 300.0, 100.0)

if st.button("Predict"):
    # Convert to DataFrame
    columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
               'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
               'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

    input_data = pd.DataFrame([[male, age, education, currentSmoker, cigsPerDay,
                                BPMeds, prevalentStroke, prevalentHyp, diabetes,
                                totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                              columns=columns)

    # Scale & PCA
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)

    # Probabilities
    rf_prob = rf_model.predict_proba(input_data_pca)[0][1]
    lr_prob = logreg_model.predict_proba(input_data_pca)[0][1]

    # Choose best model
    if rf_prob >= lr_prob:
        final_prediction = rf_model.predict(input_data_pca)[0]
        confidence = rf_prob
        model_used = "Random Forest"
    else:
        final_prediction = logreg_model.predict(input_data_pca)[0]
        confidence = lr_prob
        model_used = "Logistic Regression"

    # Display result
    st.subheader(f"🧠 Model Used: {model_used}")
    st.write(f"Prediction: {'💔 Risk of Heart Disease' if final_prediction == 1 else '✅ No Risk of Heart Disease'}")
