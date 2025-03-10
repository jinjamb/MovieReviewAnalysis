import string
import os

folder = "C:\\Users\\maxim\\Desktop\\UNIVM1S2\\NLP\\SentimentAnalysis\\smallsamplepos"
path = os.path.realpath(folder)

def clean_text(text):
    text = text.lower()
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
                #with open(f, 'w') as file:
                #    file.write(cleaned_text)