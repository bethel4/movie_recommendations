from ast import List
from csv import DictWriter, writer
import csv
import json
import time
from fastapi import FastAPI,Request
import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from pydantic import BaseModel
from typing import List
from model import movieModel
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.models import load_model

app = FastAPI()
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def movieRcommendation(model):
     movie_df = pd.read_csv("movies.csv")
     df = pd.read_csv("ratings.csv")
     users = df.userId.unique()
     movies = df.movieId.unique()
     user_id = 4562
     userid2idx = {o:i for i,o in enumerate(users)}
     movieid2idx = {o:i for i,o in enumerate(movies)}
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
     
     ratings = model.predict([user_id,movies_not_watched]).flatten()
     print(ratings)
     top_ratings_indices = ratings.argsort()[-10:]
     recommended_movie_ids = [
     movieid2idx.get(movies_not_watched[x][0]) for x in top_ratings_indices
  ]
     recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
     movies = []
     for row in recommended_movies.itertuples():
        movies.append({ 
            "title":row.title,
                   "genere" :row.genres,
                   "movieId":row.movieId
                    })
  
        return movies

@app.get('/movies')
async def root():
    jsonArray = []
    moviecatagories = []
    #read csv file
    with open('movies.csv', encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
  
    #convert python jsonArray to JSON String and write to file
 
# Filter python objects with list comprehensions
    for i in jsonArray:
        dict = {'id':i['genres'],'genres':i['genres'],'movies':[{'title':i['title'],'movieId':i['movieId']}]}
        if(i['genres']=='Comedy'):
            moviecatagories.append(dict)
        if(i['genres']=='Action'):
             moviecatagories.append(dict)
        if(i['genres']=='Drama'):
             moviecatagories.append(dict)
        if(i['genres']=='Romance'):
             moviecatagories.append(dict)
# Transform python object back into json
   
# Show json
    # print (moviecatagories)

    return moviecatagories


class Movie(BaseModel):
    movieId: int
    title: str
    genres: str 
   

@app.post('/recommendation/')
async def create_recommendation(movies:list[Movie]):
      field_names = ['userId','movieId','rating','timestamp']
      ts = time.time()
      List = [{"userId":4562, "movieId":639,"rating":5 ,"timestamp":int(ts)},
              {"userId":4562, "movieId":742, "rating":5  ,"timestamp":int(ts)},
              {"userId":4562,  "movieId":1785, "rating":5  ,"timestamp":int(ts)}]
   
      for i in List:
          json.dumps(i)
          with open('ratings.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            dictwriter_object.writerow(i)
            f_object.close()
    #   movieModel()
      
      new_model = load_model('movie_recommendation.h5')
      
      return movieRcommendation(new_model)


#  add to datasets, generate a new user id, movies

#code ur model to newfile or here after adding it