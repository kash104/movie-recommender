import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your movies dataset
movies = pd.read_csv("movies.csv")  # Change the filename if yours is different

# Clean and prepare
movies = movies[['title', 'overview']].dropna()
movies['combined'] = movies['title'] + " " + movies['overview']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined'])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# Save files
pickle.dump(movies, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Saved movies.pkl and similarity.pkl!")