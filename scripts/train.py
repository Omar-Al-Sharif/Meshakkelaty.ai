from feature_extraction import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.models import Model
import random
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class CustomSequenceGenerator(Sequence):
    def __init__(self, data_lines, batch_size):
        self.data_lines = data_lines
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data_lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_lines = self.data_lines[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, Y_batch = process_data_batch(batch_lines)
        max_seq_len = np.max([len(seq) for seq in X_batch + Y_batch])
        X_padded = pad_sequences(X_batch, max_seq_len, CHAR_TO_ID['<PAD>'])
        Y_padded = pad_sequences(Y_batch, max_seq_len, DIACRITIC_TO_ID['<PAD>'], convert_to_one_hot, len(DIACRITIC_TO_ID))
        return X_padded, Y_padded


def process_data_batch(lines_batch):
    X_batch, Y_batch = char2embeddings(lines_batch)
    return X_batch, Y_batch


def pad_sequences(sequences, max_len, padding_value, to_one_hot_func=None, num_classes=None):
    padded_sequences = []

    for seq in sequences:
        seq = list(seq)
        seq.extend([padding_value] * (max_len - len(seq)))
        padded_sequences.append(np.asarray(seq))

    if to_one_hot_func and num_classes:
        padded_sequences = [to_one_hot_func(seq, num_classes) for seq in padded_sequences]

    return np.asarray(padded_sequences)


def diacritization_model():
    input_seq = Input(shape=(None,))

    embedding_layer = Embedding(input_dim=len(CHAR_TO_ID),
                                output_dim=25,
                                embeddings_initializer=glorot_normal(seed=961))(input_seq)

    bi_lstm_layer_1 = Bidirectional(LSTM(units=256,
                                         return_sequences=True,
                                         kernel_initializer=glorot_normal(seed=961)))(embedding_layer)
    dropout_layer_1 = Dropout(0.5)(bi_lstm_layer_1)
    bi_lstm_layer_2 = Bidirectional(LSTM(units=256,
                                         return_sequences=True,
                                         kernel_initializer=glorot_normal(seed=961)))(dropout_layer_1)
    dropout_layer_2 = Dropout(0.5)(bi_lstm_layer_2)

    time_distributed_layer_1 = TimeDistributed(Dense(units=512,
                                                      activation='relu',
                                                      kernel_initializer=glorot_normal(seed=961)))(dropout_layer_2)
    time_distributed_layer_2 = TimeDistributed(Dense(units=512,
                                                      activation='relu',
                                                      kernel_initializer=glorot_normal(seed=961)))(time_distributed_layer_1)

    output_layer = TimeDistributed(Dense(units=len(DIACRITIC_TO_ID),
                                        activation='softmax',
                                        kernel_initializer=glorot_normal(seed=961)))(time_distributed_layer_2)

    diacritization_model = Model(input_seq, output_layer)
    diacritization_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return diacritization_model



def train_diacritization_model(diacritization_model, num_epochs, batch_size, training_data, validation_data):
    random.shuffle(training_data)
    training_data = list(sorted(training_data, key=lambda line: len(clean_data(line))))
    random.shuffle(validation_data)
    validation_data = list(sorted(validation_data, key=lambda line: len(clean_data(line))))
    batches_per_epoch = len(training_data) // batch_size
    diacritization_checkpoint_path = '../checkpoints/epoch{epoch:02d}.ckpt'
    diacritization_checkpoint_callback = ModelCheckpoint(filepath=diacritization_checkpoint_path, save_weights_only=True, save_freq='epoch')
    training_data_generator = CustomSequenceGenerator(training_data, batch_size)
    validation_data_generator = CustomSequenceGenerator(validation_data, batch_size)
    diacritization_model.fit(x=training_data_generator, validation_data=validation_data_generator, epochs=num_epochs, callbacks=[diacritization_checkpoint_callback])


train_data = None
with open('../data/train.txt', 'r', encoding='utf-8') as file:
    train_data = file.readlines()

validation_data = None
with open('../data/val.txt', encoding='utf-8') as file:
    validation_data = file.readlines()

model=diacritization_model()
start_time = time.time()
train_diacritization_model(model, 10, 512, wrap_lines(train_data), wrap_lines(validation_data))
end_time = time.time()
print('--- %s seconds ---' % round(end_time - start_time, 2))