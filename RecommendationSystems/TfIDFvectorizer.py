import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv("movies.csv")

# combine genres for each movei into a single string 
genres_combined = movies['genres'].str.replace('|', ' ')

# create a Tfidvectorizer object to transform the movie genres into a Tf-idf representation
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(genres_combined)


# calc cos sim
cosine_sim = cosine_similarity(tfidf_matrix)

#create a dataframe with the cosine sim scores
similarity_df = pd.Dataframe(cosine_sim, index= movies['titles'], columns= movies['titles'])


movie = input(" Enter a movie you like : ")

movie_index = similarity_df.index.get_loc(movie)

top_10 = similarity_df.iloc[movie_index].sort_values(ascending=False)[1:11]

print(f'Top 10 movies similar to {movie}')
print(top_10)


