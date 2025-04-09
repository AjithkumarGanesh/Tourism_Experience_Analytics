import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from PIL import Image

# --- Sidebar image ---
with st.sidebar:
    img = Image.open("D:/Tourism_Experience_Analytics/streamlit/tourism_banner.jpg.jpeg")
    st.image(img, use_container_width=True)
    st.markdown("### Tourism Experience Insights")

# --- CSS Animation for smooth fade-in ---
st.markdown("""
    <style>
    .stApp {
        animation: fadeInAnimation 1.2s ease-in;
    }
    @keyframes fadeInAnimation {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Load model using joblib
model = joblib.load("classification_xgb_visitmode_model.pkl")

# Load feature columns using pickle
with open("classification_feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Load label encoder using joblib
label_encoder = joblib.load("classification_visitmode_label_encoder.pkl")

# Custom mapping from encoded value to detailed visit mode info
visit_mode_map = {
    0: ("01", "Business"),
    1: ("02", "Couples"),
    2: ("03", "Family"),
    3: ("04", "Friends"),
    4: ("05", "Solo")
}

# Title
st.title("Visit Mode Prediction Dashboard")
st.markdown("Predict **Visit Mode** by selecting values for each input feature.")

# Load reference data to populate dropdowns
try:
    reference_df = pd.read_csv("test123.csv")
except Exception as e:
    st.warning(f"Could not load reference data: {e}")
    reference_df = pd.DataFrame(columns=feature_columns)

# User input collection
user_input = {}
for col in feature_columns:
    unique_values = reference_df[col].dropna().unique()
    if len(unique_values) > 0:
        user_input[col] = st.selectbox(f"{col}", sorted(unique_values.astype(str)))
    else:
        user_input[col] = st.text_input(f"{col} (enter manually)")

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Match datatypes if reference_df is available
for col in feature_columns:
    if col in reference_df.columns:
        input_df[col] = input_df[col].astype(reference_df[col].dtype)

# Predict and display visit mode
if st.button("Predict Visit Mode"):
    try:
        prediction_encoded = model.predict(input_df[feature_columns])[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        visit_code, visit_mode = visit_mode_map[prediction_encoded]

        st.success("Prediction Successful!")
        st.markdown(f"""
        ### Predicted Visit Mode Details:
        - **Code:** `{visit_code}`
        - **Visit Mode:** `{visit_mode}`
        """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
