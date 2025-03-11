import string
import os

folders = ["..\\SentimentAnalysis\\smallsamplepos", "..\\SentimentAnalysis\\smallsampleneg"]

pathpos = os.path.realpath(folders[0])
pathneg = os.path.realpath(folders[1])

cleanedsamplepos = os.makedirs("..\\SentimentAnalysis\\smallsamplepos\\cleaned", exist_ok=True)
cleanedsampleneg = os.makedirs("..\\SentimentAnalysis\\smallsampleneg\\cleaned", exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = text.replace('<br />', '\n')
    text = text.replace('é', 'e')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

if __name__ == "__main__":
    for folder in folders:
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            if os.path.isfile(f):
                print(f)
                print("\n")
                with open(f, 'r') as file:
                    text = file.read()
                    cleaned_text = clean_text(text)
                    print(cleaned_text)
                    print("\n")
                    if "pos" in f:
                        with open(os.path.join("..\\SentimentAnalysis\\smallsamplepos\\cleaned", filename), 'w') as cleaned_file:
                            cleaned_file.write(cleaned_text)
                    else:
                        with open(os.path.join("..\\SentimentAnalysis\\smallsampleneg\\cleaned", filename), 'w') as cleaned_file:
                            cleaned_file.write(cleaned_text)
                    