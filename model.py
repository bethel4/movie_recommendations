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
    df['rating'] = df['rating'].fillna(0)
    users = df.userId.unique()
    movies = df.movieId.unique()
    userid2idx = {o:i for i,o in enumerate(users)}
    movieid2idx = {o:i for i,o in enumerate(movies)}
    num_users = len(userid2idx)
    num_movies = len(movieid2idx)
    n_latent_factors=64
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
                              epochs =10, validation_data = ([valid.userId,valid.movieId],valid.rating),
                              verbose = 1)
    model.save('movie_recommendation.h5')
    
    