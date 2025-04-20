import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

vectorized_dir = ".\\vectorized"

def load_vectorized_data(directory):
    """Charge les données d'entraînement et de test préalablement vectorisées."""
    train_data = np.load(os.path.join(directory, 'train_data.npz'), allow_pickle=True)
    test_data = np.load(os.path.join(directory, 'test_data.npz'), allow_pickle=True)
    
    X_train = train_data['X'].item() if isinstance(train_data['X'], np.ndarray) else train_data['X']
    y_train = train_data['y']
    X_test = test_data['X'].item() if isinstance(test_data['X'], np.ndarray) else test_data['X']
    y_test = test_data['y']

    print(f"Type de X_train: {type(X_train)}")
    
    with open(os.path.join(directory, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    return X_train, y_train, X_test, y_test, vectorizer

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Entraîne et évalue plusieurs modèles de classification."""
    
    models = {
        'Régression Logistique': LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'),
        'SVM Linéaire': LinearSVC(C=1.0, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'Naive Bayes': MultinomialNB(alpha=0.1)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\n [TRAIN] Entraînement du modèle: {name}")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Négatif', 'Positif'])
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Précision: {accuracy:.4f}")
        print(report)
        print("Matrice de confusion:")
        print(conf_matrix)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return results, best_model

def visualize_results(results):
    """Visualise les résultats des différents modèles."""
    accuracies = [result['accuracy'] for result in results.values()]
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Modèle')
    plt.ylabel('Précision')
    plt.title('Comparaison des modèles de classification')
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vectorized_dir, 'model_comparison.png'))
    plt.show()

def save_best_model(model, vectorizer, directory):
    """Sauvegarde le meilleur modèle."""
    with open(os.path.join(directory, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"Meilleur modèle sauvegardé dans {os.path.join(directory, 'best_modelk.pkl')}")

def predict_example(model, vectorizer, text):
    """Prédit le sentiment d'un exemple de texte."""
    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]
    probability = None

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_tfidf)[0]

    sentiment = "positif" if prediction == 1 else "négatif"
    print(f"Sentiment prédit: {sentiment}")
    if probability is not None:
        print(f"Probabilité: Négatif {probability[0]:.4f}, Positif {probability[1]:.4f}")
    
    return sentiment, probability

if __name__ == "__main__":
    print("[LOADING] Chargement des données vectorisées...")
    X_train, y_train, X_test, y_test, vectorizer = load_vectorized_data(vectorized_dir)
    
    print(f"Nombre d'exemples d'entraînement: {X_train.shape[0]}")
    print(f"Nombre d'exemples de test: {X_test.shape[0]}")
    print(f"Nombre de caractéristiques: {X_train.shape[1]}")
    
    print("\n[TRAIN] Entraînement des modèles...")
    results, best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    visualize_results(results)

    best_model_name = [name for name, result in results.items() 
                       if result['model'] is best_model][0]
    
    print(f"\nMeilleur modèle: {best_model_name} avec une précision de {results[best_model_name]['accuracy']:.4f}")
    save_best_model(best_model, vectorizer, vectorized_dir)