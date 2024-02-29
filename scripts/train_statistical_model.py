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
        
def closestWords(possibilities_set, misspelled_word):
    '''
    This function returns a list of the closest words to the misspelled word
    '''
    # check the minimum edit distance between the misspelled word and the possibilities
    min_distance = float("inf")
    closest_words = []

    for word in possibilities_set:
        distance = edit_distance(misspelled_word[:-1], word[:-1])

        if distance < min_distance:
            # Found a new minimum distance, update the list of closest words
            min_distance = distance
            closest_words = [word]
        elif distance == min_distance:
            # Found a tie, add this word to the list of closest words
            closest_words.append(word)

    return closest_words

def wordEndings(word):
    '''
    This function returns the number of diacritics on the last letter of the word
    '''
    count = 0
    if bool(re.match(r"[\u064b-\u0652]{2}", word[-2:])):
        count = 2
    elif bool(re.match(r"[\u064b-\u0652]", word[-1])):
        count = 1
    return count

def matchLastDiacritic(closest_word, misspelled_word):
    '''
    This function matches the last diacritic of the misspelled word with the closest word
    '''
    count_endings_misspelled = wordEndings(misspelled_word)
    count_endings_closest = wordEndings(closest_word)
    # if both words have no diacritics on the last letter, return the closest word
    if count_endings_misspelled == 0 and count_endings_closest == 0:
        most_probable_word = closest_word
    
    # if the misspelled word has no diacritics on the last letter, return the closest word with the last diacritic of the misspelled word
    elif count_endings_misspelled == 0:
        most_probable_word = closest_word[:-count_endings_closest]
    
    # if the closest word has no diacritics on the last letter, return the closest word with the last diacritic of the misspelled word
    elif count_endings_closest == 0:
        most_probable_word = closest_word + misspelled_word[-count_endings_misspelled:]
    
    # if both words have diacritics on the last letter, return the closest word with the last diacritic of the misspelled word
    else:
        most_probable_word = closest_word[:-count_endings_closest] + misspelled_word[-count_endings_misspelled:]
    return most_probable_word

def post_process_model(word_dict, unigram, misspelled_word):
    '''
    This function takes a dictionary of undiacritized words as keys and all possible diacritized words as values
    in addition to a unigram frequency distribution of the words in the file
    and a misspelled word as input and returns the most probable diacritized word
    '''
    undiacritized = remove_diacritics(misspelled_word)
    if undiacritized in word_dict:
        possibilities_set = word_dict[undiacritized]
        closest_words = closestWords(possibilities_set, misspelled_word)
        if len(closest_words) == 1:
            return matchLastDiacritic(closest_words[0], misspelled_word)
        else:
            most_frequent_word = max(closest_words, key=lambda word: unigram[word])
            if sum(1 for word in closest_words if unigram[word] == unigram[most_frequent_word]) > 1:
                most_frequent_word = random.choice([word for word in closest_words if unigram[word] == unigram[most_frequent_word]])
            return matchLastDiacritic(most_frequent_word, misspelled_word)
    else:
        return misspelled_word


def import_statistical_model(model_path):
    '''
    This function imports a pickle file and returns a dictionary of words
    along with a frequency distribution unigram model
    '''
    with open(model_path, 'rb') as pickle_file:
        word_dict, unigram = pkl.load(pickle_file)
    return word_dict, unigram

if __name__ == '__main__':
    
    train_statistical_model('../data/train.txt', "./utilities/pickle_files/STATISTICAL_MODEL.pickle")

    

    
