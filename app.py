import streamlit as st
import pandas as pd
import pickle
import requests

# Load your data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Your OMDb API key
OMDB_API_KEY = '235121d6'  # ← Replace with your actual key

def fetch_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    poster = data.get('Poster', '')
    plot = data.get('Plot', 'No description available')
    year = data.get('Year', '')
    rating = data.get('imdbRating', '')
    return poster, plot, year, rating

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movie_list:
        title = movies.iloc[i[0]].title
        poster, plot, year, rating = fetch_movie_details(title)
        recommended.append((title, poster, plot, year, rating))
    return recommended

# Streamlit UI
st.title("Movie Recommender System")

movie_name = st.selectbox(
    "Choose a movie you like",
    movies['title'].values
)

if st.button('Show Recommendations'):
    recommendations = recommend(movie_name)
    for title, poster, plot, year, rating in recommendations:
        st.markdown(f"### {title} ({year})")
        st.image(poster, width=200)
        st.markdown(f"*IMDb Rating:* {rating}")
        st.markdown(f"*Plot:* {plot}")
        st.markdown("---")