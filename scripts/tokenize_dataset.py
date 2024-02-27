import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import pickle as pkl

class TashkeelDataset(Dataset):
    def __init__(self, name, data):
        self.name = name
        self._load_dicts()
        if isinstance(data, str):  
            with open(data, 'r', encoding='utf-8') as file:
                self.lines = list(tqdm(file, f"Reading {self.name} Lines"))
            self.tokenized_lines = self._tokenize_lines()
            self.embedded_data = self._embedd_lines()
        else:  
            self.embedded_data = data

    def __len__(self):
        return len(self.embedded_data)

    def __getitem__(self, idx):
        x, y = self.embedded_data[idx]
        return torch.tensor(x).to(device), torch.tensor(y).to(device)
    
    def _remove_tashkeel(self,data):
        #double damma, double fatha, double kasera, damma, fatha, kasera, sukoon, shadd
        TASHKEEL_SET = {'ٌ', 'ً', 'ٍ', 'ُ', 'َ', 'ِ', 'ْ', 'ٌّ', 'ّ'}
        DIACRITICS_REGEX = re.compile('|'.join(TASHKEEL_SET))
        return re.sub(DIACRITICS_REGEX, '', data)
    
    def _one_hot_encode(self, indices, size):
        return [[1 if i == elem else 0 for i in range(size)] for elem in indices]
    
    def _chunk_text(self, text, chunk_size):
        chunks = []
        words = re.findall(r'\S+', text)

        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk += f"{word} "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = f"{word} "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return list(filter(None, chunks))
    
    def _tokenize_lines(self):
        # Define a pattern to match specific punctuation marks
        punctuation_pattern1 = r'([.,:;؛)\]}»،])'
        punctuation_pattern2 = r'([(\[{«])'
        tokenized_lines = []

        for line in tqdm(self.lines, f"Tokenizing {self.name} Lines"):
            # Replace matched punctuation marks with the same followed by a line break
            splitted_line = re.sub(punctuation_pattern1, r'\1\n', line)
            splitted_line = re.sub(punctuation_pattern2, r'\n\1', splitted_line)

            # Further split the splitted line into substrings based on line breaks
            for sub_line in splitted_line.split('\n'):
                cleaned_sub_line = self._remove_tashkeel(sub_line).strip()
                if 0 < len(cleaned_sub_line) <= 500:
                    tokenized_lines.append(sub_line.strip())

                elif len(cleaned_sub_line) > 500:
                    tokenized_lines.extend(self._chunk_text(sub_line.strip(), 500))
    
        return tokenized_lines

    def _load_dicts(self):
        with open( '../utilities/pickle_files/LETTERS.pickle', 'rb') as file:
            self.LETTERS = pkl.load(file)
        with open( '../utilities/pickle_files/DIACRITICS.pickle', 'rb') as file:
            self.DIACRITICS = pkl.load(file)
        with open( '../utilities/pickle_files/CHAR_TO_ID.pickle', 'rb') as file:
            self.CHAR_TO_ID = pkl.load(file)
        with open( '../utilities/pickle_files/DIACRITIC_TO_ID.pickle', 'rb') as file:
            self.DIACRITIC_TO_ID = pkl.load(file)
        
    def _embedd_lines(self):
        inputs_embeddings=[]
        for line in tqdm(self.tokenized_lines, f"Embedding {self.name} Lines"):
            x = [self.CHAR_TO_ID['<SOS>']]
            y = [self.DIACRITIC_TO_ID['<SOS>']]

            for index, char in enumerate(line):
                if char in self.DIACRITICS:
                    continue

                if char in self.CHAR_TO_ID:
                    x.append(self.CHAR_TO_ID[char])
                else:
                    x.append(self.CHAR_TO_ID['<UNK>'])

                if char not in self.LETTERS:
                    y.append(self.DIACRITIC_TO_ID[''])

                else:
                    char_diac = ''
                    if index + 1 < len(line) and line[index + 1] in self.DIACRITICS:
                        char_diac = line[index + 1]
                        if index + 2 < len(line) and line[index + 2] in self.DIACRITICS and char_diac + line[index + 2] in self.DIACRITIC_TO_ID:
                            char_diac += line[index + 2]
                        elif index + 2 < len(line) and line[index + 2] in self.DIACRITICS and line[index + 2] + char_diac in self.DIACRITIC_TO_ID:
                            char_diac = line[index + 2] + char_diac
                    y.append(self.DIACRITIC_TO_ID[char_diac])

            x.append(self.CHAR_TO_ID['<EOS>'])
            y.append(self.DIACRITIC_TO_ID['<EOS>'])
            y = self._one_hot_encode(y, len(self.DIACRITIC_TO_ID))
            
            inputs_embeddings.append((x, y)) 
            
        return inputs_embeddings
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TashkeelDataset('train dataset','../data/val.txt')
    val_dataset = TashkeelDataset('validation dataset','../data/val.txt')
    
    # Save the tensors separately
    torch.save(train_dataset.embedded_data, open('../data/train_data.pt', 'wb+'))
    torch.save(val_dataset.embedded_data, open('../data/val_data.pt', 'wb+'))

    print("The processed dataset is saved as tensors.")

