import nltk
import os

folders = ["..\\SentimentAnalysis\\smallsamplepos\\cleaned", "..\\SentimentAnalysis\\smallsampleneg\\cleaned"]
tokenized_folders = ["..\\SentimentAnalysis\\tokenizedsamplepos", "..\\SentimentAnalysis\\tokenizedsampleneg"]

os.makedirs(tokenized_folders[0], exist_ok=True)
os.makedirs(tokenized_folders[1], exist_ok=True)

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

if __name__ == "__main__":
    for folder, tokenized_folder in zip(folders, tokenized_folders):
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            if os.path.isfile(f):
                with open(f, 'r') as file:
                    text = file.read()
                    tokenized_text = tokenize_text(text)
                    with open(os.path.join(tokenized_folder, filename), 'w') as tokenized_file:
                        tokenized_file.write(tokenized_text)