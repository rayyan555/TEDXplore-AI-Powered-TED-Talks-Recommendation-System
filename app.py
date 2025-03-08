import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load TED Talks dataset
@st.cache_data
def load_data():
    return pd.read_csv("tedx_dataset.csv")

data = load_data()

# Preprocess text data (remove NaN values in 'details')
data["details"] = data["details"].fillna("")

# Convert text descriptions to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["details"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_talks(title, num_recommendations=5):
    """Function to recommend TED Talks based on content similarity"""
    if title not in data["title"].values:
        return None  # Return None if the title is not in the dataset

    idx = data[data["title"] == title].index[0]  # Find the index of the selected talk
    sim_scores = list(enumerate(cosine_sim[idx]))  # Get similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]  # Sort by similarity

    talk_indices = [i[0] for i in sim_scores]  # Get indices of similar talks
    return data.iloc[talk_indices][["title", "main_speaker", "details"]]  # Return recommended talks

# Streamlit UI
st.title("🎤 TED Talks Recommendation System")
st.write("Find similar TED Talks based on content!")

# User input
selected_talk = st.selectbox("Choose a TED Talk:", data["title"].unique())

if st.button("Recommend"):
    recommendations = recommend_talks(selected_talk)
    
    if recommendations is not None:
        st.subheader("Recommended TED Talks:")
        for i, row in recommendations.iterrows():
            st.write(f"**{row['title']}** - {row['main_speaker']}")
            st.write(f"📜 {row['details'][:150]}...")  # Show a preview of the description
            st.write("---")
    else:
        st.write("No recommendations found. Try another talk!")
