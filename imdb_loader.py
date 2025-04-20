import kagglehub
import os
import shutil 
import pandas as pd
import tempfile

base_dir = os.path.join(".", "imdb_data")

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
    
pos_dir = os.path.join(base_dir, "pos")
neg_dir = os.path.join(base_dir, "neg")

for directory in (pos_dir, neg_dir):
    os.makedirs(directory, exist_ok=True)

csv_path = os.path.join("..", "archive", "IMDB_Dataset.csv")

"""
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
data_dir = path

csv_path = os.path.join(data_dir, "IMDB Dataset.csv")

csv_path = os.path.join("..", "archive", "IMDB_Dataset.csv")
dataf = pd.read_csv(csv_path)

for idx, row in dataf.iterrows():
    sentiment = row["sentiment"]
    text = row["review"]
    filename = f"{sentiment}_{idx}.txt"
    target = pos_dir if sentiment == "positive" else neg_dir
    with open(os.path.join(target, filename), "w", encoding="utf-8") as f:
        f.write(text)
"""

dataf = pd.read_csv(csv_path)

data_pos = dataf[dataf["sentiment"] == "positive"]
data_neg = dataf[dataf["sentiment"] == "negative"]

for idx, row in data_pos.iterrows():
    filename = f"pos_{idx}.txt"
    with open(os.path.join(pos_dir, filename), "w", encoding="utf-8") as f:
        f.write(row["review"])

for idx, row in data_neg.iterrows():
    filename = f"neg_{idx}.txt"
    with open(os.path.join(neg_dir, filename), "w", encoding="utf-8") as f:
        f.write(row["review"])

n_pos = (dataf["sentiment"] == "positive").sum()
n_neg = (dataf["sentiment"] == "negative").sum()
print(f"Nombre de critiques positives : {n_pos}")
print(f"Nombre de critiques négatives : {n_neg}")
print(f"Total des critiques : {len(dataf)}")
print("Dataset IMDb téléchargé et décompressé avec succès !")