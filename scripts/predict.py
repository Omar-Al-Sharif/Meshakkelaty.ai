import torch
import pickle as pkl
import numpy as np
from tokenize_dataset import TashkeelDataset
from train_neural_model import MeshakkelatyModel
from train_statistical_model import import_statistical_model, post_process_model

TEST_INPUT_PATH = '../data/test_input.txt'
TEST_OUTPUT_PATH = '../data/test_output.txt'

# Load the ID_TO_DIACRITIC mapping
with open('../utilities/pickle_files/ID_TO_DIACRITIC.pickle', 'rb') as file:
    ID_TO_DIACRITIC = pkl.load(file)

def neural_model_inference(tashkeel_dataset: TashkeelDataset, model):

    # Predict diacritics using the model
    with torch.no_grad():
        model.eval()
        predicted_diacritics = model(tashkeel_dataset.embedded_data).squeeze()[1:]

    # Initialize an empty string to store the final output
    output_line = ''

    # Iterate through each character and its corresponding prediction
    for original_char, diacritic_prediction in zip(tashkeel_dataset._remove_tashkeel(tashkeel_dataset.tokenized_lines), predicted_diacritics):
        # Append the original character to the output
        output_line += original_char
        # Skip further processing for non-letter characters
        if original_char not in tashkeel_dataset.LETTERS:
            continue
        # Skip diacritics containing '<'
        if '<' in ID_TO_DIACRITIC[np.argmax(diacritic_prediction.cpu().numpy())]:
            continue
        # Append the predicted diacritic to the output
        output_line += ID_TO_DIACRITIC[np.argmax(diacritic_prediction.cpu().numpy())]
    return output_line


def statistical_model_inference(input, word_dict, unigram):

    words = input.split(' ')
    improved_tashkeel = ''
    for word in words:
        corrected_word = post_process_model(word_dict, unigram, word)
        improved_tashkeel += f' {corrected_word}'
    
    return improved_tashkeel
 
    


if __name__ == '__main__':

    test_dataset = TashkeelDataset('test dataset', TEST_INPUT_PATH)
    meshakkelaty = MeshakkelatyModel(test_dataset.CHAR_TO_ID, test_dataset.DIACRITIC_TO_ID)
    checkpoint_path = f'../checkpoints/checkpoint_latest.pt'
    # Load the previously saved weights
    latest_checkpoint = torch.load(checkpoint_path)
    meshakkelaty.load_state_dict(latest_checkpoint['model_state_dict'])

    neural_model_predictions = neural_model_inference(test_dataset, meshakkelaty)

    statistical_model_dict, statistical_model_unigram = import_statistical_model("./utilities/pickle_files/STATISTICAL_MODEL.pickle")
    statistical_model_predictions = statistical_model_inference(neural_model_predictions, statistical_model_dict, statistical_model_unigram)

    with open(TEST_OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
        file.write(statistical_model_predictions)






    


    
    
