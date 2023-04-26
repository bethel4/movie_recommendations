from fastapi import FastAPI
import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from pydantic import BaseModel
import typing
app = FastAPI()
df = pd.read_csv("ratings.csv")
movie_df = pd.read_csv("movies.csv")
model = load_model('movie_recommendation.h5')
users = df.userId.unique()
movies = df.movieId.unique()
userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}
num_users = len(userid2idx)
num_movies = len(movieid2idx)
@app.get('/')
async def root():
    return {"example": "this is a test","data":0}

def recommend_movie(df,movie_df,movieid2idx,userid2idx):  
  
  # Let us get a user and see the top recommendations.
  user_id = df.userId.sample(1).iloc[0]
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


class Movie(BaseModel):
    title: str
    genre: str 
    movie_id: int

@app.post('/recommendation/')
async def create_recommendation(movies:list[Movie]):
      # recommend_movie(df,movie_df,movieid2idx,userid2idx)

      return recommend_movie(df,movie_df,movieid2idx,userid2idx)


#  add to datasets, generate a new user id, movies

#code ur model to newfile or here after adding it