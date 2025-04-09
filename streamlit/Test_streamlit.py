import streamlit as st
import pandas as pd
import joblib
import time

# Load trained LightGBM model
model = joblib.load("best_lgbm_model.pkl")

# Load dataset for unique values
df = pd.read_csv("test123.csv")

# Drop rows with missing target to match training behavior
df = df.dropna(subset=["Rating_Scaled"])

st.set_page_config(page_title="Tourism Rating Predictor", layout="wide")
st.title("Tourism Rating Prediction Dashboard")

# ---- Display image on the left ----
col1, col2 = st.columns([1, 3])

with col1:
    st.image(r"D:/Tourism_Experience_Analytics/streamlit/tourism_banner.jpg.jpeg", use_container_width=True)


with col2:
    st.markdown("<h4 style='animation: pulse 2s infinite; color: #4B8BBE;'>Provide the following inputs to predict the <strong>scaled rating</strong> (0 to 1).</h4>", unsafe_allow_html=True)

# ---- Extract unique values dynamically from dataset ----
user_ids = sorted(df['UserId'].dropna().unique())
visit_years = sorted(df['VisitYear'].dropna().unique())
visit_months = sorted(df['VisitMonth'].dropna().unique())
visit_modes = sorted(df['VisitMode'].dropna().unique())
attraction_ids = sorted(df['AttractionId'].dropna().unique())
attraction_city_ids = sorted(df['AttractionCityId'].dropna().unique())
attraction_types = sorted(df['AttractionType'].dropna().unique())
user_continent_ids = sorted(df['User_ContinentId'].dropna().unique())
user_country_ids = sorted(df['User_CountryId'].dropna().unique())
countries = sorted(df['Country'].dropna().unique())
continents = sorted(df['Continent'].dropna().unique())
avg_rating_values = sorted(df['AvgRatingPerMode'].dropna().unique())

# ---- Dropdown Inputs ----
UserId = st.selectbox("User ID", user_ids)
VisitYear = st.selectbox("Visit Year", visit_years)
VisitMonth = st.selectbox("Visit Month", visit_months)
VisitMode = st.selectbox("Visit Mode", visit_modes)
AttractionId = st.selectbox("Attraction ID", attraction_ids)
AttractionCityId = st.selectbox("Attraction City ID", attraction_city_ids)
AttractionType = st.selectbox("Attraction Type", attraction_types)
User_ContinentId = st.selectbox("User Continent ID", user_continent_ids)
User_CountryId = st.selectbox("User Country ID", user_country_ids)
Country = st.selectbox("User Country", countries)
Continent = st.selectbox("User Continent", continents)
AvgRatingPerMode = st.selectbox("Average Rating per Mode", avg_rating_values)

# ---- Encoders (learned from dataset itself) ----
visit_mode_map = {v: i for i, v in enumerate(df['VisitMode'].dropna().unique())}
continent_map = {v: i for i, v in enumerate(df['Continent'].dropna().unique())}
attraction_type_map = {v: i for i, v in enumerate(df['AttractionType'].dropna().unique())}
country_map = {v: i for i, v in enumerate(df['Country'].dropna().unique())}

# ---- Derived Encoded Fields ----
VisitMode_Encoded = visit_mode_map[VisitMode]
Continent_Encoded = continent_map[Continent]
AttractionType_Encoded = attraction_type_map[AttractionType]
Country_Encoded = country_map[Country]
AvgRatingPerMode_Scaled = round(AvgRatingPerMode / 5.0, 4)

# ---- Construct Input DataFrame ----
input_data = pd.DataFrame([{
    'UserId': UserId,
    'VisitYear': VisitYear,
    'VisitMonth': VisitMonth,
    'VisitMode': VisitMode,
    'AttractionId': AttractionId,
    'AttractionCityId': AttractionCityId,
    'AttractionType': AttractionType,
    'User_ContinentId': User_ContinentId,
    'User_CountryId': User_CountryId,
    'Country': Country,
    'Continent': Continent,
    'VisitMode_Encoded': VisitMode_Encoded,
    'Continent_Encoded': Continent_Encoded,
    'AttractionType_Encoded': AttractionType_Encoded,
    'AvgRatingPerMode': AvgRatingPerMode,
    'AvgRatingPerMode_Scaled': AvgRatingPerMode_Scaled
}])

# ---- Replace object columns with encoded versions ----
input_data["VisitMode"] = VisitMode_Encoded
input_data["Continent"] = Continent_Encoded
input_data["AttractionType"] = AttractionType_Encoded
input_data["Country"] = Country_Encoded

# ---- Predict Button ----
if st.button("Predict Scaled Rating"):
    with st.spinner("Making prediction... please wait"):
        time.sleep(1.5)  # just for animation
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Scaled Rating: **{prediction:.4f}**")
            rating_1_to_5 = round(prediction * 5, 2)
            st.info(f"Predicted Rating (1–5 scale): **{rating_1_to_5} / 5**")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
