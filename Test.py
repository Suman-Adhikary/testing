######################################################## Import Packages ########################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import sys, pathlib
import requests
import os


api_key = "125157ce4628811990b663910aad700e"



######################################################### Import Dataset #########################################################

current_directory = pathlib.Path(__file__).parent.absolute()

# Define the relative paths to the CSV and pickle files
csv_file_path = current_directory / "Dataset" / "Hollywood_with_Poster.csv"
pickle_file_path = current_directory / "Dataset" / "Hollywood.pkl"

# Read the CSV and pickle files using the relative paths
TMDB_DATA = pd.read_csv(csv_file_path)
SIMILARITY_PICKLE = pd.read_pickle(pickle_file_path)





################################################### STOPWORD AND TRANSFORMATION ##################################################

tfidf = TfidfVectorizer(stop_words='english')
TMDB_DATA['Combined'] = TMDB_DATA['Combined'].fillna('')
tfidf_matrix = tfidf.fit_transform(TMDB_DATA['Combined'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



########################################################## RECOMMENDATION ########################################################

@st.cache_data
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = SIMILARITY_PICKLE[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:35]
    movie_indices = [i[0] for i in sim_scores]
    return TMDB_DATA[['imdb_id', 'original_title', 'Poster']].iloc[movie_indices]



######################################################### CREATE A MOVIE LIST ####################################################

