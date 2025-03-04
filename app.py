import gradio as gr
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_heart_disease(
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_scaled)[:, 1]  # Get probability of having heart disease
    probability_percent = round(prediction_proba[0] * 100, 2)  # Convert to percentage
    result = f"Probability of Heart Disease: {probability_percent}%"
    return result

iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age (e.g., 45)"),
        gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Radio([0, 1, 2, 3], label="Chest Pain Type (0–3)"),
        gr.Number(label="Resting Blood Pressure (e.g., 130)"),
        gr.Number(label="Cholesterol Level (e.g., 200)"),
        gr.Radio([0, 1], label="Fasting Blood Sugar (0 or 1)"),
        gr.Radio([0, 1, 2], label="Resting ECG Results (0–2)"),
        gr.Number(label="Max Heart Rate Achieved (e.g., 150)"),
        gr.Radio([0, 1], label="Exercise-Induced Angina (0 or 1)"),
        gr.Number(label="Oldpeak (ST Depression Induced by Exercise) (e.g., 1.5)"),
        gr.Radio([0, 1, 2], label="Slope of ST Segment (0–2)"),
        gr.Radio([0, 1, 2, 3], label="Number of Major Vessels Colored by Fluoroscopy (0–3)"),
        gr.Radio([0, 1, 2, 3], label="Thalassemia Type (0–3)"),
    ],
    outputs=gr.Textbox(label="Prediction Probability"),
    title="Heart Disease Prediction",
    description="Enter health parameters to get the probability of heart disease.",
)

iface.launch()
