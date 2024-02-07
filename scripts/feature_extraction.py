import numpy as np
import pickle as pkl
from scripts.pre_process import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from tensorflow.keras.layers import Input

def get_max(lines):
      X,Y = char2embeddings(lines)
      X_max_seq_len = np.max([len(x) for x in X])
      return X_max_seq_len


def TF_IDF(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    inputs = Input(shape=(get_max(lines), tfidf_matrix.shape[1]))  # Input shape is now TF-IDF features
    return inputs
     
with open( '../utilities/pickle_files/LETTERS.pickle', 'rb') as file:
    LETTERS = pkl.load(file)
with open( '../utilities/pickle_files/DIACRITICS.pickle', 'rb') as file:
    DIACRITICS = pkl.load(file)
with open( '../utilities/pickle_files/CHAR_TO_ID.pickle', 'rb') as file:
    CHAR_TO_ID = pkl.load(file)
with open( '../utilities/pickle_files/DIACRITIC_TO_ID.pickle', 'rb') as file:
    DIACRITIC_TO_ID = pkl.load(file)

CHAR_TO_ID['<UNK>'] = len(CHAR_TO_ID)

def char2embeddings(data):
    for line in data:
        x = [CHAR_TO_ID['<SOS>']]
        y = [DIACRITIC_TO_ID['<SOS>']]

        for index, char in enumerate(line):
            if char in DIACRITICS:
                continue
            if char in CHAR_TO_ID:
              x.append(CHAR_TO_ID[char])
            else: 
              x.append(CHAR_TO_ID['<UNK>'])

            if char not in LETTERS:
                y.append(DIACRITIC_TO_ID[''])
            else:
                char_diac = ''
                if index + 1 < len(line) and line[index + 1] in DIACRITICS:
                    char_diac = line[index + 1]
                    if index + 2 < len(line) and line[index + 2] in DIACRITICS and char_diac + line[index + 2] in DIACRITIC_TO_ID:
                        char_diac += line[index + 2]
                    elif index + 2 < len(line) and line[index + 2] in DIACRITICS and line[index + 2] + char_diac in DIACRITIC_TO_ID:
                        char_diac = line[index + 2] + char_diac
                y.append(DIACRITIC_TO_ID[char_diac])

        x.append(CHAR_TO_ID['<EOS>'])
        y.append(DIACRITIC_TO_ID['<EOS>'])
        y = convert_to_one_hot(y, len(DIACRITIC_TO_ID))
        inputs_embeddings.append(x)
        one_hot_labels.append(y)

    inputs_embeddings = np.asarray(inputs_embeddings)
    one_hot_labels = np.asarray(one_hot_labels)

    return inputs_embeddings, one_hot_labels