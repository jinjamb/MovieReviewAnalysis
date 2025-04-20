import string
import os

#folders = [".\\smallsamplepos", ".\\smallsampleneg"]


#cleanedsamplepos = os.makedirs(".\\smallsamplepos\\cleaned", exist_ok=True)
#cleanedsampleneg = os.makedirs(".\\smallsampleneg\\cleaned", exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = text.replace('<br />', ' ')
    text = text.replace('é', 'e')
    text = text.translate(str.maketrans('', '', string.punctuation))
    accents = {
        'à': 'a', 'â': 'a', 'ä': 'a',
        'è': 'e', 'ê': 'e', 'ë': 'e',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c'
    }
    for accent, replacement in accents.items():
        text = text.replace(accent, replacement)

    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def run_cleaner():
    folders = [".\\imdb_data\\pos", ".\\imdb_data\\neg"]

    for folder in folders:
        os.makedirs(os.path.join(folder, "cleaned"), exist_ok=True)

    pathpos = os.path.realpath(folders[0])
    pathneg = os.path.realpath(folders[1])

    for folder in folders:
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            if os.path.isfile(f):
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(f, 'r', encoding=encoding) as file:
                            text = file.read()
                            break
                    except UnicodeDecodeError:
                        if encoding == 'cp1252':
                            print(f"Impossible de lire {f}, utilisation de 'replace'")
                            with open(f, 'r', encoding='utf-8', errors='replace') as file:
                                text = file.read()
                        else:
                            continue
                cleaned_text = clean_text(text)
                if "pos" in f:
                    output_path = os.path.join(".\\imdb_data\\pos\\cleaned", filename)
                else:
                    output_path = os.path.join(".\\imdb_data\\neg\\cleaned", filename)
                with open(output_path, 'w', encoding='utf-8') as cleaned_file:
                    cleaned_file.write(cleaned_text)
                    