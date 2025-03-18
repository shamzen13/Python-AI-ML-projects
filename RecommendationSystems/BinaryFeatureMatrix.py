

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")


#create binary feature matrix

genre_matrix = pd.get_dummies(movies['genres'].str.split("|").apply(pd.Series).stack())

# this splits the genres (in a row) into coloumns and then into long data format and finally into a binary feature matrix

# computing similarity matrix
similarity = cosine_similarity(genre_matrix)

#function to get rec movies

def get_recommendations(title, top_n= 5):

    # find index of movie with given title 
    idx = movies[movies['title'] == title].index[0]

    # get score similarity 
    similarity_scores = list(enumerate(similarity[idx]))

    # sort in desc order
    similarity_scores = sorted(similarity_scores, key=lambda x:x[1], reverse= True)

    # get top_n indeces
    movies_indices = [i[0] for i in similarity_scores[1:top_n+1]]

    return movies['title'].iloc[movies_indices]


# ask user for movie name
title = input("enter the title of your favorite movie : ")

print("Top 5 similar movies : ")
print(get_recommendations(title))

