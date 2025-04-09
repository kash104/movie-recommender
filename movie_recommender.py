import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movies = pd.read_csv('movies_metadata.csv', low_memory=False)

# Drop rows with missing 'overview'
movies = movies[movies['overview'].notnull()]

# Reset index after cleaning
movies = movies.reset_index()

#new line to reduce size
movies = movies.sample(1000, random_state=42).reset_index(drop=True)

#new show sample titles
print("some available movie titles in your sample:")
print(movies['title'].head(20))

# TF-IDF Vectorizer for 'overview' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found. Check spelling or try another."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example
movie_name = "Silenced"
print(f"\nTop 5 recommendations for '{movie_name}':")
recommendations = get_recommendations(movie_name)
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")