import nltk
import random
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

#pos_folder = ".\\smallsamplepos\\cleaned" 
#neg_folder = ".\\smallsampleneg\\cleaned"

pos_folder = ".\\imdb_data\\pos\\cleaned" 
neg_folder = ".\\imdb_data\\neg\\cleaned"

output_dir = ".\\vectorized"

os.makedirs(output_dir, exist_ok=True)

def load_data_from_folder(folder, label):
    texts = []
    labels = []
    
    try:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        text = file.read()
                        texts.append(text)
                        labels.append(label)
                except UnicodeDecodeError:
                    with open(filepath, 'r', encoding='latin-1') as file:
                        text = file.read()
                        texts.append(text)
                        labels.append(label)
    except FileNotFoundError:
        print(f"ATTENTION: Le dossier {folder} n'existe pas.")
    
    return texts, labels

def create_tfidf_vectors(pos_folder, neg_folder, ngram_range=(1, 3), min_df=5, max_features=20000):
    print("Chargement des critiques positives...")
    pos_texts, pos_labels = load_data_from_folder(pos_folder, 1)
    print(f"Nombre de critiques positives chargées: {len(pos_texts)}")
    
    print("Chargement des critiques négatives...")
    neg_texts, neg_labels = load_data_from_folder(neg_folder, 0)
    print(f"Nombre de critiques négatives chargées: {len(neg_texts)}")

    all_texts = pos_texts + neg_texts
    all_labels = pos_labels + neg_labels
    
    print(f"Total des critiques: {len(all_texts)}")

    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        use_idf=True,
        lowercase=True,
        stop_words='english'
    )

    print("Vectorisation des données d'entraînement...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print("Vectorisation des données de test...")
    X_test_tfidf = vectorizer.transform(X_test)
    print("Sauvegarde des données vectorisées...")
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    np.savez(os.path.join(output_dir, 'train_data.npz'),
             X=X_train_tfidf,
             y=np.array(y_train))
    
    np.savez(os.path.join(output_dir, 'test_data.npz'),X=X_test_tfidf,y=np.array(y_test))
    
    print(f"Nombre de caractéristiques: {X_train_tfidf.shape[1]}")
    print(f"Nombre d'exemples d'entraînement: {X_train_tfidf.shape[0]}")
    print(f"Nombre d'exemples de test: {X_test_tfidf.shape[0]}")
    vocabulary_size = len(vectorizer.vocabulary_)
    print(f"Taille du vocabulaire: {vocabulary_size}")
    
    return vectorizer, X_train_tfidf, y_train, X_test_tfidf, y_test

if __name__ == "__main__":
    vectorizer, X_train, y_train, X_test, y_test = create_tfidf_vectors(
        pos_folder=pos_folder, 
        neg_folder=neg_folder,
        ngram_range=(1, 2),
        min_df=5,
        max_features=20000
    )

    print("[CHECK] Vecteurs TF-IDF créés et sauvegardés avec succès.")

    features=vectorizer.get_feature_names_out()

    pos_texts, _ = load_data_from_folder(pos_folder, 1)
    neg_texts, _ = load_data_from_folder(neg_folder, 0)
    #sample_docs = random.sample(pos_texts, min(10, len(pos_texts))) + random.sample(neg_texts, min(10, len(neg_texts)))
    #sample_docs = random.sample(pos_texts,10) + random.sample(neg_texts,10)
    '''
    print("\n--- Visualisation TF‑IDF sur 10 exemples pos + 10 exemples neg ---")
    for i, doc in enumerate(sample_docs, 1):
        vec = vectorizer.transform([doc])
        tokens_scores = sorted(zip(vec.indices, vec.data),key=lambda x: x[1],reverse=True)[:10]
        print(f"\nDocument #{i} (label={'pos' if i<=10 else 'neg'}):")
        for idx, score in tokens_scores:
            print(f"  {features[idx]:<20s} → {score:.3f}")
    '''

    '''
    Test de classification avec Logistic Regression en utilisant les probabilités fournies par le modèle
    Cependant, ce n'est pas la meilleure approche pour le problème de classification car même si le modèle prédit une 
    probabilité importante pour une review, l'utilisateur qui a écrit la review n'a pas forcément apprécié le film comme
    le modèle le prédit. 
    Essai avec VADER de nltk.sentiment pour voir si le modèle est plus efficace que le modèle de classification
    

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    X_sample = vectorizer.transform(sample_docs)
    probability = classifier.predict_proba(X_sample)

    print("\n--- Probabilités par document (nég vs pos) ---")
    for i, (p_neg, p_pos) in enumerate(probability, 1):
        print(f"Review: {sample_docs[i-1]}")
        if (p_neg > 0.8):
            print("1 star")
        elif (p_pos > 0.8):
            print("5 star")
        elif (p_neg > 0.6 and p_pos < 0.4):
            print("2 star")
        elif (p_pos > 0.6 and p_neg < 0.4):
            print("4 star")
        else:
            print("3 star")
    '''