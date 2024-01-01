import pyarabic.araby as parb
import pandas as pd
import re
import json
import pickle as pkl
import os

"""
    pdargs arguments put 1 in front the one you want
    raw_data: get the raw data without preprocessing
    data_symbols_removed: remove symbols
    normalized_with_diacritics: normalizes diacritics
    normalized_without_diacritics: removes diactrics
"""
"""
    normalization_args put 1 in front of the one you want, these are extra normalization steps
    if none of them are selected, tatweel, hamza and alef lam are the only ones normalized
    normalized_alef: normalizes all alefs to ا
    nomralized_teh: normalizes ة to ه
"""

def preprocess_dataset(file,pdargs=[],normalization_args=[]):
    # if no files are required to be written, skip all their blocks
    if type(pdargs) == list:
        pdargs = {
            "raw_data": 0,
            "data_symbols_removed": 0,
            "normalized_with_diacritics": 0,
            "normalized_without_diacritics": 0,
        }
    
    if type(normalization_args) == list:
        normalization_args = {
            "normalized_alef": 0,
            "normalized_teh": 0,
        }
        
    
    # Reorganizing data into a more readable form
    with open(file,"r",encoding='utf-8') as f:
        lines = f.read()
    f.close()
    train_string="".join(lines)
    raw_data = train_string.split('.')
    
    if pdargs["raw_data"] == 1:
        raw_data_split = pd.DataFrame(raw_data)
        raw_data_split.to_json('./output/raw_data/raw_data.json')
        raw_data_split.to_csv('./output/raw_data/raw_data.csv')
        
    # removing punctuation, numbers and symbols
    data_symbols_removed = []
    for sentence in raw_data:
        symbol_regex = r"[^\u0600-\u06FF\u0640\u064B-\u065F ]"
        symbol_result = re.sub(symbol_regex,"",sentence)
        no_comma = re.sub("\u060c","",symbol_result)
        no_comma = no_comma.translate(str.maketrans('', '', '؛؟‘~,.":/[]×<>!'))
        data_symbols_removed.append(re.sub(" +"," ",no_comma))
    
    if pdargs["data_symbols_removed"] == 1:
        symbols_removed_df = pd.DataFrame(data_symbols_removed)
        symbols_removed_df.to_json("./output/symbols_removed/symbols_removed.json")
        symbols_removed_df.to_csv("./output/symbols_removed/symbols_removed.csv")
    
    # removing tatweel and normalize hamzat and ligatures
    normalized_with_diacritics = []
    for sentence in data_symbols_removed:
        no_tatweel = parb.strip_tatweel(sentence)
        normamlized_ligature = parb.normalize_ligature(no_tatweel)
        if normalization_args["normalized_alef"] == 1:
            normamlized_ligature = parb.normalize_alef(normamlized_ligature)
        if normalization_args["normalized_teh"] == 1:
            normamlized_ligature = parb.normalize_teh(normamlized_ligature)
        normalized_hamza = parb.normalize_hamza(normamlized_ligature)
        normalized_with_diacritics.append(normalized_hamza)
    
    if pdargs["normalized_with_diacritics"] == 1:
        normalized_with_diacritics_df = pd.DataFrame(normalized_with_diacritics)
        normalized_with_diacritics_df.to_json("./output/normalized_with_diacritics/normalized_with_diacritics.json")
        normalized_with_diacritics_df.to_csv("./output/normalized_with_diacritics/normalized_with_diacritics.csv")
    
    
    # removing diacritics
    normalized_without_diacritics=[]
    for sentence in normalized_with_diacritics:
        no_diacritics = parb.strip_harakat(sentence)
        no_shadda = parb.strip_shadda(no_diacritics)
        normalized_without_diacritics.append(no_shadda)
        
    if pdargs["normalized_without_diacritics"]:
        normalized_without_diacritics_df = pd.DataFrame({"sentences":normalized_with_diacritics})
        normalized_without_diacritics_df.to_json("./output/normalized_without_diacritics/normalized_without_diacritics.json")
        normalized_without_diacritics_df.to_csv("./output/normalized_without_diacritics/normalized_without_diacritics.csv")
    
    #extracting Lables for training
    diacritic_label = []
    for i in range(0,len(normalized_with_diacritics)):
        set_without_diacritics = set(normalized_without_diacritics[i])
        difference = ""
        for char in normalized_with_diacritics[i]:
            if char not in set_without_diacritics:
                difference += char
        diacritic_label.append(difference)
        
    labled_dataset = []
    for i in range(0,len(normalized_without_diacritics)):
        input_output_dict ={"sentence":normalized_without_diacritics[i],"diacritic":diacritic_label[i]}
        labled_dataset.append(input_output_dict)



    if not os.path.exists("./output/"):
        os.mkdir(os.path.join( os.getcwd(),"output"))
    
    with open('./output/labled_dataset.jsonl',"+w") as f:
        for data in labled_dataset:
            json.dump(data,f)
            f.write("\n")
            
    with open('./output/labled_dataset.txt',"+w", encoding="utf-8") as f:
        for data in labled_dataset:
            f.write(data['sentence'] + "\n")
            
    labled_dataset_df = pd.DataFrame(labled_dataset)
    #labled_dataset_df.drop("Unnamed: 0",axis=1)
    labled_dataset_df.to_csv('./output/labled_dataset.csv')
        
    
    
    
