import data_cleaner
import tokenizer
import model
import imdb_loader
import tokenizer
import tester
import load_nltk_reviews
from print_color import print

import os
import shutil
import pandas as pd

imdb_loader.load_imdb()
print("Dataset loaded successfully.", tag="CHECK", tag_color='green', color="green")
print("-"*50, color="white")
data_cleaner.run_cleaner()
print("Data cleaning completed.", tag="CHECK", tag_color='green', color="green")
print("-"*50, color="white")
tokenizer.run_tokenizer()
print("Tokenization completed.", tag="CHECK", tag_color='green', color="green")
print("-"*50, color="white")
model.train_model()
print("Model training complete.", tag="CHECK", tag_color='green', color="green")
print("-"*50, color="white")
load_nltk_reviews.main()
print("NLTK reviews to test complete.", tag="CHECK", tag_color='green', color="green")
print("-"*50, color="white")
tester.main()