import nltk
from nltk.corpus import movie_reviews
import os
import numpy as np
import pandas as pd



def load_movie_reviews():
    
    reviews = []
    for fileid in movie_reviews.fileids():
        label = movie_reviews.categories(fileid)[0]
        reviews.append((movie_reviews.raw(fileid), label))
    
    output_dir = "movie_reviews_texts"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (review, label) in enumerate(reviews):
        with open(os.path.join(output_dir, f"review_{idx}_{label}.txt"), "w", encoding="utf-8") as f:
            f.write(review)

def main():
    nltk.download('movie_reviews')
load_movie_reviews()