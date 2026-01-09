import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("gb_model.pkl")

st.set_page_config(
    page_title="F1 Racing Classification",
    page_icon="🏎️",
    layout="centered"
)

st.title("🏎️ F1 Racing Outcome Prediction")
st.write("This app predicts race classification using a Gradient Boosting model.")

# Example inputs (adjust according to your features)
speed = st.number_input("Average Speed", min_value=0.0)
lap_time = st.number_input("Lap Time", min_value=0.0)
pit_stops = st.number_input("Pit Stops", min_value=0, step=1)

if st.button("Predict"):
    input_data = np.array([[speed, lap_time, pit_stops]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Class: {prediction[0]}")

st.markdown("### 🔍 Input Race Data")
col1, col2 = st.columns(2)

with col1:
    speed = st.number_input("Average Speed")

with col2:
    lap_time = st.number_input("Lap Time")

pit_stops = st.slider("Number of Pit Stops", 0, 10)


st.info("This model was trained using Gradient Boosting Classifier for academic purposes.")
