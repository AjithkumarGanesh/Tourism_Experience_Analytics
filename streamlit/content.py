import streamlit as st
import pandas as pd
import pickle

# Sidebar with image
with st.sidebar:
    st.image("D:/Tourism_Experience_Analytics/streamlit/tourism_banner.jpg.jpeg", caption="Travel Recommender", width=180)

# Load model
with open("D:/Tourism_Experience_Analytics/streamlit/content_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Load data
df = pd.read_csv("D:/Tourism_Experience_Analytics/streamlit/test123.csv")

# Set page title
st.title("Content-Based Attraction Recommender")

# Dropdowns
user_ids = df['UserId'].unique()
attraction_ids = df['AttractionId'].unique()

# Select user and recommendation limit
selected_user = st.selectbox("Select User ID", sorted(user_ids))
recommendation_limit = st.selectbox("Select Number of Recommendations", [5, 10, 15, 20])

# Button to trigger recommendations
if st.button("Get Recommendations"):
    # Get list of visited attractions for the selected user
    visited_attractions = df[df['UserId'] == selected_user]['AttractionId'].tolist()

    # Get the list of recommended attractions (this part should be replaced by your trained model's logic)
    recommended_attractions = df[df['AttractionId'].isin(visited_attractions)].head(recommendation_limit)

    # Columns to display
    columns_to_display = ['AttractionId', 'AttractionType', 'Country', 'Continent', 'Rating_Scaled']
    
    # Display recommendations as a table
    st.success(f"Top {recommendation_limit} Recommended Attractions for User {selected_user}:")
    st.table(recommended_attractions[columns_to_display])  # Display as a table
    
    # Balloon animation on successful recommendation
    st.balloons()
