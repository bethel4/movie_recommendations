from ast import List
from csv import DictWriter, writer
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
 
app = FastAPI()
@app.get('/')
async def root():
    return {"example": "this is a test","data":0}


class Movie(BaseModel):
    movieId: int
    title: str
    genres: str 
   

@app.post('/recommendation/')
async def create_recommendation(movies:list[Movie]):
      field_names = ['userId','movieId','rating','timestamp']
      ts = time.time()
      List = [{"userId":4562, "movieId":5532,"rating":5 ,"timestamp":int(ts)},
              {"userId":4562, "movieId":7845, "rating":5  ,"timestamp":int(ts)},
              {"userId":4562,  "movieId":1785, "rating":5  ,"timestamp":int(ts)}]
   
      for i in List:
          json.dumps(i)
          with open('ratings.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            dictwriter_object.writerow(i)
            f_object.close()
      movie_field = ['movieId','title','genres']
      for i in movies:
        with open('movies.csv','a') as f_objects:
              movie_dict = DictWriter(f_objects,fieldnames=movie_field)
              movie_dict.writerow({"movieId":i.movieId,"title":i.title,"genres":i.genres})
              f_objects.close()
      response =   movieModel()
      return response


#  add to datasets, generate a new user id, movies

#code ur model to newfile or here after adding it