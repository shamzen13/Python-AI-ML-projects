import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")


# extract movie titles and genres into seperate lists
titles = movies['title'].tolist()
genres = movies['genres'].str.split("|").tolist()



# create a bags of words representations of the movie genres
def create_bow(genre_list):
    bow = {}
    for genre in genre_list:
        bow[genre] = 1
    return bow


# create a list of bow representations
bag_of_words = [create_bow(movie_genres) for movie_genres in genres]

# create a dataframe to store the bow rep of movie genres
genre_df = pd.DataFrame(bag_of_words, index=titles).fillna(0)

cosine_sim = cosine_similarity(genre_df)

#create a df for the cos_sim
similarity_df = pd.DataFrame(cosine_sim, index=genre_df, columns=genre_df.index)


movie = input("Enter a movie you like : ")
movie_index = similarity_df.index.get_loc(movie)

top_10 = similarity_df.iloc[movie_index].sort_values(ascending=False)[1:11]

print(f'Top 10 movies similar to {movie}')
print(top_10)


