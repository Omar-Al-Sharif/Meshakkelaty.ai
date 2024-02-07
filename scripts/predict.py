from train import *
import os
import tensorflow as tf


with open(  '../utilities/pickle_files/ID_TO_DIACRITIC.pickle', 'rb') as file:
    ID_TO_DIACRITIC = pkl.load(file)

def inference(input_line, model):
    # Convert the input line into embeddings
    input_embeddings, _ = char2embeddings([input_line])
    # Predict diacritics using the model
    predicted_diacritics = model.predict(input_embeddings).squeeze()[1:]
    # Initialize an empty string to store the final output
    output_line = ''
    # Iterate through each character and its corresponding prediction
    for original_char, diacritic_prediction in zip(clean_data(input_line), predicted_diacritics):
        # Append the original character to the output
        output_line += original_char
        # Skip further processing for non-letter characters
        if original_char not in LETTERS:
            continue
        # Skip diacritics containing '<'
        if '<' in ID_TO_DIACRITIC[np.argmax(diacritic_prediction)]:
            continue
        # Append the predicted diacritic to the output
        output_line += ID_TO_DIACRITIC[np.argmax(diacritic_prediction)]
    return output_line


model = diacritization_model()
checkpoint_path = "../checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

# Load the previously saved weights
model.load_weights(latest_checkpoint)

print(inference('مرحبا كيف الحال يا صديقي العزيز', model))