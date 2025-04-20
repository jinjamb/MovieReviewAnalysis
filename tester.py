import pickle
import os
import numpy as np
import pandas as pd
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from print_color import print

model_path = ".\\vectorized\\best_model.pkl"
vectorizer_path = ".\\vectorized\\tfidf_vectorizer.pkl"


def sentiment_shades(review):
    sia = SentimentIntensityAnalyzer()
    results = []
    for text in review:
        coumpound_score = sia.polarity_scores(text)["compound"]
        if coumpound_score >= 0.6:
            label = "5 Stars"
        elif coumpound_score >= 0.2:
            label = "4 Stars"
        elif coumpound_score >= -0.2:
            label = "3 Stars"
        elif coumpound_score >= -0.6:
            label = "2 Stars"
        else:
            label = "1 Star"
        results.append((label, coumpound_score))
    return results

def predict_sentiment(text):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    """Prédit le sentiment d'un texte."""
    text_tfidf = vectorizer.transform([text])
    
    prediction = model.predict(text_tfidf)[0]
    
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_tfidf)[0]
    
    sentiment = "positif" if prediction == 1 else "négatif"
    print(f"Texte: \"{text[:300]}...\"")
    print(f"Sentiment prédit: {sentiment}")
    
    if probability is not None:
        print("Probabilité: ")
        print(f"Négatif {probability[0]:.4f}", color="red")
        print(f"Positif {probability[1]:.4f}", color="green")

    if probability[0] > 0.6:
        print("1 Star", tag="NOTATION", tag_color='yellow', color="white")
    elif probability[0] > 0.4 and probability[0] <= 0.6:
        print("2 Stars", tag="NOTATION", tag_color='yellow', color="white")
    elif probability[0] > 0.2 and probability[0] <= 0.4:
        print("3 Stars", tag="NOTATION", tag_color='yellow', color="white")
    elif probability[0] <= 0.2 and probability[0] > 0.1:
        print("4 Stars", tag="NOTATION", tag_color='yellow', color="white")
    elif probability[0] <= 0.1:
        print("5 Stars", tag="NOTATION", tag_color='yellow', color="white") 

    print("-" * 50)
    return sentiment, probability

"""
reviews = ["This movie was fantastic!", "I did not like this movie at all.", 
           "The plot was very interesting and engaging.", "The acting was terrible and the script was boring.",
           "I would highly recommend this film to anyone.", "It was a waste of time."]
"""
def take_random_reviews():
    reviews = os.listdir("movie_reviews_texts")

    pos_files = [r for r in reviews if "pos" in r]
    neg_files = [r for r in reviews if "neg" in r]
    selected_pos = random.sample(pos_files, 10)
    selected_neg = random.sample(neg_files, 10)
    positive_reviews = []
    negative_reviews = []

    for review in selected_pos + selected_neg:
        with open(f"movie_reviews_texts/{review}", 'r', encoding='utf-8') as f:
            content = f.read()
            predict_sentiment(content)
            if "pos" in review:
                positive_reviews.append(content)
            else:
                negative_reviews.append(content)

    print("\n10 Positive Reviews:")
    for pos_review in positive_reviews:
        predict_sentiment(pos_review)
        print("-" * 50)

    print("\n10 Negative Reviews:")
    for neg_review in negative_reviews:
        predict_sentiment(neg_review)
        print("-" * 50)

"""

for review in reviews:
    predict_sentiment(review)
"""

def test_interactive():
    print("\nTESTEUR DE SENTIMENT DE CRITIQUES DE FILMS")
    print("Entrez 'exit' pour quitter")
    print("-" * 50)
    
    while True:
        user_input = input("\nEntrez une critique de film: ")
        if user_input.lower() == 'exit':
            break
        predict_sentiment(user_input)

def main():
    take_random_reviews()
    test_interactive()