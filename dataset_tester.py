import pandas as pd

# Load the dataset
file_path = r'..\archive\IMDB_Dataset.csv'
data = pd.read_csv(file_path)

for sentiment in data['sentiment'][:25000]:
    print(sentiment)