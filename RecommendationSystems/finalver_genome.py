import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

genome_tags = pd.read_csv("genome-tags.csv")
genome_scores = pd.read_csv("genome-scores.csv")


# merge the genome scores df with the genome tags df to get the relavence score
merged_genome = genome_scores.merge(genome_tags, on='tagId', how='left')


#filter the df to only include the top 20 tags with the highest scores
top_tags  = merged_genome[merged_genome['relavance'] > 0.5]
top_tags.reset_index(drop=True, inplace= True)

# group the top df by 'movieId' and join the 'tag' values in the 'tag' column seperated by comma
grouped_tags = top_tags.groupby('movieId')['tag'].apply(lambda x : ' , '.join(x)).reset_index()

#merge the 'movies' df with the grouped_tags df
final_df = movies.merge(grouped_tags, on='movieId', how = 'left')

# select only desired columns in the final df
final_df = final_df[['movieId', 'title', 'genres', 'tag']]

def add_genres_to_tag(row):
    if pd.isnull(row['tag']):
        return row['genres'].replace("|", ",")
    else:
        return row['tag'] + "," + row['genres'].replace("|", ",")
    

final_df['tag'] = final_df.apply(lambda row: add_genres_to_tag(row), axis = 1)

#extract the movie titles and tags into seperate lists

titles = final_df['titles'].tolist()
tags = final_df['tag'].str.split(",").tolist()


#create bow rep 

def create_bow(tag_list):
    bow = {}
    if not isinstance(tag_list, float):
        for tag in tag_list:
            bow[tag] = 1
    return bow

# create a bow list
bag_of_words = [create_bow(movie_tags) for movie_tags in tags]

# create a df to store the bow rep of movie tags
tag_df = pd.DataFrame(bag_of_words, index=titles).fillna(0)

cos_sim = cosine_similarity(tag_df)

similarity_df = pd.DataFrame(cosine_similarity, index=tag_df.index, columns=tag_df.index)


movie = input('Enter a movie you like: ')


movie_index = similarity_df.index.get_loc(movie)


top_10 = similarity_df.iloc[movie_index].sort_values(ascending=False)[1:11]


print(f'Top 10 similar movies to {movie}:')
print(top_10)