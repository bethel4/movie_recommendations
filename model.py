import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense,Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model

from keras.layers import dot
# specifically for deeplearning.
from keras.layers import Dropout, Flatten,Activation,Input,Embedding
import random as rn

def movieModel():
    df = pd.read_csv("ratings.csv")
    movie_df = pd.read_csv("movies.csv")
    df['rating'] = df['rating'].fillna(0)
    users = df.userId.unique()
    movies = df.movieId.unique()
    userid2idx = {o:i for i,o in enumerate(users)}
    movieid2idx = {o:i for i,o in enumerate(movies)}
    num_users = len(userid2idx)
    num_movies = len(movieid2idx)
    n_latent_factors=64
    df['userId'] = df['userId'].apply(lambda x: userid2idx[x])
    df['movieId'] = df['movieId'].apply(lambda x: movieid2idx[x])
    split = np.random.rand(len(df)) < 0.9
    train = df[split]
    valid = df[~split]
    user_input = Input(shape=(1,),name = 'User_Input',dtype = 'int64' )
    user_embedding=Embedding(num_users,n_latent_factors,name='user_embedding')(user_input)
    user_vec = Flatten(name = 'FlattenUsers')(user_embedding)
    movie_input = Input(shape=(1,),name = 'Movie_Input',dtype = 'int64')
    movie_embedding = Embedding(num_movies,n_latent_factors,name = 'movie_embedding')(movie_input)
    movie_vec = Flatten(name = 'FlattenMovies')(movie_embedding)
    sim = dot([user_vec,movie_vec],name = 'Similarty-Dot-Product',axes = 1)
    model = keras.models.Model([user_input,movie_input],sim)
    model.compile(optimizer=Adam(learning_rate=1e-3),loss='mse')
    History = model.fit([train.userId,train.movieId],train.rating, batch_size=64,
                              epochs =20, validation_data = ([valid.userId,valid.movieId],valid.rating),
                              verbose = 1)
    user_id = df.userId.sample(1).iloc[0]
  # user_id = 4452
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
 ]["movieId"]
    movies_not_watched = list(
    set(movies_not_watched).intersection(set(movieid2idx.keys()))
 )
    movies_not_watched = [[movieid2idx.get(x)] for x in movies_not_watched]
    user_encoder = userid2idx.get(user_id)  
    user_id = [[user_encoder]] * len(movies_not_watched)
    movies_not_watched = np.array(movies_not_watched)
    user_id = np.array(user_id)
    ratings = model.predict([np.array(user_id),np.array(movies_not_watched)]).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
    movieid2idx.get(movies_not_watched[x][0]) for x in top_ratings_indices
  ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
  # print([recommended_movies])
  # return recommended_movies
    movies = []
    for row in recommended_movies.itertuples():
        movies.append({ 
            "title":row.title,
                   "genere" :row.genres,
                   "movieId":row.movieId
                    })
  
    return movies