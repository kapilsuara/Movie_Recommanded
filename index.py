import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies_modified.csv")
    credits = pd.read_csv("tmdb_5000_credits_modified.csv")  # Load the modified credits file
    movies = movies.merge(credits, on="id")  # Merge using 'id'
    return movies

movie_data = load_data()

# Data Preprocessing
def preprocess_data(movie_data):
    # Select relevant columns
    movies = movie_data[["genres", "title_x", "id", "keywords", "overview", "director", "top_5_cast", "popularity", "release_date", "revenue"]]
    movies.dropna(inplace=True)

    # Categorize popularity
    high_threshold = movies['popularity'].quantile(0.75)
    medium_threshold = movies['popularity'].quantile(0.5)
    low_threshold = movies['popularity'].quantile(0.25)

    def categorize_popularity(popularity):
        if popularity >= high_threshold:
            return 'High'
        elif popularity >= medium_threshold:
            return 'Medium'
        elif popularity >= low_threshold:
            return 'Low'
        else:
            return 'Very Low'

    movies['popularity_category'] = movies['popularity'].apply(categorize_popularity)
    movies.drop("popularity", axis=1, inplace=True)

    # Categorize release date
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

    def categorize_release_date(release_date):
        if pd.isnull(release_date):
            return 'Unknown'
        elif release_date.year < 1980:
            return 'Classic'
        elif release_date.year < 2000:
            return 'Old'
        elif release_date.year < 2015:
            return 'Recent'
        else:
            return 'New'

    movies['release_date_category'] = movies['release_date'].apply(categorize_release_date)
    movies.drop("release_date", axis=1, inplace=True)

    # Categorize revenue
    blockbuster_threshold = movies['revenue'].quantile(0.75)
    hit_threshold = movies['revenue'].quantile(0.5)
    moderate_threshold = movies['revenue'].quantile(0.25)

    def categorize_revenue(revenue):
        if revenue >= blockbuster_threshold:
            return 'Blockbuster'
        elif revenue >= hit_threshold:
            return 'Hit'
        elif revenue >= moderate_threshold:
            return 'Moderate'
        else:
            return 'Low'

    movies['revenue_category'] = movies['revenue'].apply(categorize_revenue)
    movies.drop("revenue", axis=1, inplace=True)

    # Convert genre, keywords, and director columns
    def convert_to_list(obj):
        l = []
        for item in ast.literal_eval(obj):
            l.append(item["name"])
        return l

    movies["genres"] = movies["genres"].apply(convert_to_list)
    movies["keywords"] = movies["keywords"].apply(convert_to_list)
    movies["top_5_cast"] = movies["top_5_cast"].apply(lambda x: x if isinstance(x, list) else [])  # Ensure it's a list

    # Create 'tags' column
    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["tags"] = movies[["genres", "keywords", "overview", "director", "top_5_cast", "popularity_category", "release_date_category", "revenue_category"]].apply(
        lambda row: ' '.join(' '.join(map(str, col)) if isinstance(col, list) else str(col) for col in row if col is not None),
        axis=1
    )
    
    new_df = movies[["title_x", "id", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

    # Apply stemming
    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])
    
    new_df["tags"] = new_df["tags"].apply(stem)

    # Feature extraction
    cv = CountVectorizer(stop_words="english", max_features=5000)
    vector = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vector)
    
    return new_df, similarity

new_df, similarity = preprocess_data(movie_data)

# List of unique movie titles for the suggestion feature
movie_titles = new_df['title_x'].unique()

# Recommendation function
def recommend(movie, new_df, similarity):
    if movie not in new_df['title_x'].values:
        st.write("Movie not found in the database.")
        return
    index = new_df[new_df['title_x'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [new_df.iloc[i[0]].title_x for i in distances[1:10]]
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")
st.write("### Find movies similar to your favorite!")

movie_name = st.selectbox("Enter a movie name:", options=movie_titles)
if st.button("Recommend"):
    if movie_name:
        recommendations = recommend(movie_name, new_df, similarity)
        if recommendations:
            st.write("Movies recommended for you:")
            for rec_movie in recommendations:
                st.write(f"- {rec_movie}")
    else:
        st.write("Please enter a movie name to get recommendations.")
