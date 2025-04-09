import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# Load Data
df = pd.read_csv("D:/Tourism_Experience_Analytics/test123.csv")

# Rename rating column if necessary
if 'Rating_Scaled' in df.columns:
    df.rename(columns={'Rating_Scaled': 'Rating'}, inplace=True)

# Load Trained Model
model = joblib.load("D:/Tourism_Experience_Analytics/streamlit/collab_model.pkl")

# Banner Image
banner = Image.open("D:/Tourism_Experience_Analytics/streamlit/tourism_banner.jpg.jpeg")
st.image(banner, caption="Tourism Recommender", width=200, use_container_width=False)

# App Title
st.title("Tourist Attraction Recommender System")

# Sidebar Inputs
user_list = df['UserId'].unique().tolist()
selected_user = st.selectbox("Select User", user_list)

top_n = st.selectbox("Select Number of Recommendations", [3, 5, 10, 15, 20], index=1)

if st.button("Submit"):
    with st.spinner("Generating Recommendations..."):
        # Pivot table for matrix factorization
        user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating', fill_value=0)
        
        # Transform user & item features
        user_features = model.transform(user_item_matrix)
        item_features = model.components_

        user_index = list(user_item_matrix.index).index(selected_user)
        user_vector = user_features[user_index]
        
        # Predicted scores
        predicted_scores = np.dot(user_vector, item_features).flatten()

        # Get already visited
        seen_items = user_item_matrix.loc[selected_user][user_item_matrix.loc[selected_user] > 0].index.tolist()
        unseen_mask = ~user_item_matrix.columns.isin(seen_items)
        unseen_items = user_item_matrix.columns[unseen_mask]
        unseen_scores = predicted_scores[unseen_mask]

        # Scale scores back to 1–5
        scaler = MinMaxScaler(feature_range=(1, 5))
        scaled_scores = scaler.fit_transform(unseen_scores.reshape(-1, 1)).flatten()

        # Create results DataFrame
        results_df = pd.DataFrame({
            'AttractionId': unseen_items,
            'Predicted Rating (1 to 5)': scaled_scores
        })

        # Merge attraction type
        results_df = results_df.merge(df[['AttractionId', 'AttractionType']].drop_duplicates(), on='AttractionId')

        # Sort and get top N
        top_recommendations = results_df.sort_values(by='Predicted Rating (1 to 5)', ascending=False).head(top_n)

        st.success("Here are your top recommendations:")
        st.dataframe(top_recommendations.reset_index(drop=True), use_container_width=True)
