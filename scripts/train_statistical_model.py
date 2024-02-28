import re
import pickle as pkl
import random
from nltk import edit_distance, FreqDist

#double damma, double fatha, double kasera, damma, fatha, kasera, sukoon, shadd
TASHKEEL_SET = {'ٌ', 'ً', 'ٍ', 'ُ', 'َ', 'ِ', 'ْ', 'ٌّ', 'ّ'}
DIACRITICS_REGEX = re.compile('|'.join(TASHKEEL_SET))
def remove_diacritics(data):
    return re.sub(DIACRITICS_REGEX, '', data)

def preprocess(file):
    '''
    Preprocess the data by removing all non arabic characters and diacritics
    '''
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read()

    pattern = r'\u000a+' 
    result = re.sub(pattern, ' ', lines) # remove end lines
    pattern = r'[^\u0621-\u0655 ]+' 
    result = re.sub(pattern, '', result) # remove non arabic characters
    pattern = r'\s+'
    result = re.sub(pattern, ' ', result) # remove extra spaces

    return result

def train_statistical_model(input_file, model_path):
    '''
    This function takes a file path as input and returns
    a dictionary of undiacritized words as keys and all possible diacritized words as values
    in addition to a unigram frequency distribution of the words in the file
    '''
    # Dictionary to store undiacritized words as keys
    # and diacritized words as values in a set
    word_dict = {}

    # Split the line into words based on spaces
    words = preprocess(input_file).split()

    unigram = FreqDist(words)

    # Process each word
    for word in words:
        # Remove diacritics using the custom function
        undiacritized_word = remove_diacritics(word)
        if undiacritized_word == word:
            continue

        # Add the undiacritized word to the dictionary
        # If the undiacritized word is not in the dictionary, create a new entry
        # Otherwise, update the existing entry with the diacritized word
        if undiacritized_word not in word_dict:
            word_dict[undiacritized_word] = {word}
        else:
            word_dict[undiacritized_word].add(word)
    with open(model_path, 'wb') as pickle_file:
        # Dump both dictionaries into the pickle file
        pkl.dump((word_dict, unigram), pickle_file)
        
    return word_dict, unigram

def import_statistical_model(model_path):
    '''
    This function imports a pickle file and returns a dictionary of words
    along with a frequency distribution unigram model
    '''
    with open(model_path, 'rb') as pickle_file:
        word_dict, unigram = pkl.load(pickle_file)
    return word_dict, unigram

if __name__ == '__main__':
    
    word_dict, unigram = train_statistical_model('../data/train.txt', "./utilities/STATISTICAL_MODEL.pickle")

    

    
