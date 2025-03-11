import string
import os

folder = "..\\SentimentAnalysis\\smallsamplepos"
path = os.path.realpath(folder)
cleanedsamplepos = os.makedirs("..\\SentimentAnalysis\\smallsamplepos\\cleaned", exist_ok=True)
cleanedsampleneg = os.makedirs("..\\SentimentAnalysis\\smallsampleneg\\cleaned", exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = text.replace('<br />', '\n')
    text = text.replace('é', 'e')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

if __name__ == "__main__":
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
                cleaned_filename = f"cleaned_{filename}"
                if "pos" in f:
                    with open(os.path.join("..\\SentimentAnalysis\\smallsamplepos\\cleaned", cleaned_filename), 'w') as cleaned_file:
                        cleaned_file.write(cleaned_text)
                else:
                    with open(os.path.join("..\\SentimentAnalysis\\smallsampleneg\\cleaned", cleaned_filename), 'w') as cleaned_file:
                        cleaned_file.write(cleaned_text)
                    