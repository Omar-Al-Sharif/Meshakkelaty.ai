import torch
from torch.utils.data import DataLoader
from tokenize_dataset import TashkeelDataset  
import torch.nn as nn 
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim 
from torchmetrics import Accuracy
from tqdm import tqdm 

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_padded = rnn_utils.pad_sequence(x_batch, batch_first=True, padding_value=train_dataset.CHAR_TO_ID['<PAD>'])
    y_padded = rnn_utils.pad_sequence(y_batch, batch_first=True, padding_value=train_dataset.DIACRITIC_TO_ID['<PAD>'])
    return x_padded, y_padded

class MeshakkelatyModel(nn.Module):
    def __init__(self, char_to_id, diacritic_to_id):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(char_to_id),
            embedding_dim=25,
            padding_idx=char_to_id['<PAD>']  
        )
        self.lstm1 = nn.LSTM(
            input_size=25,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.5,
            batch_first=True  
        )
        self.linear1 = nn.Linear(2*256, 512)
        self.linear2 = nn.Linear(512, len(diacritic_to_id))

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tensors from the saved files
    train_data = torch.load('../data/train_data.pt')
    val_data = torch.load('../data/val_data.pt')

    # Create instances of TashkeelDataset using the loaded tensors
    train_dataset = TashkeelDataset('train dataset', train_data)
    val_dataset = TashkeelDataset('validation dataset', val_data)
    
    # Create a DataLoader instance with collate_fn
    dataloader_train = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    meshakkelaty = MeshakkelatyModel(train_dataset.CHAR_TO_ID, train_dataset.DIACRITIC_TO_ID).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meshakkelaty.parameters())
    metric = Accuracy(task="multiclass", num_classes=len(train_dataset.DIACRITIC_TO_ID)).to(device)
    
    
    epochs = 10

    for epoch in range(epochs):
        meshakkelaty.train()

        # Initialize variables to accumulate correct and total predictions
        total_correct = 0
        total_samples = 0

        epoch_progress = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs}")

        for x_batch, y_batch in epoch_progress:
        
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = meshakkelaty(x_batch)
            loss = criterion(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()

            # Convert one-hot encoded predictions and targets to class indices
            y_pred_class = y_pred.argmax(dim=-1)
            y_batch_class = y_batch.argmax(dim=-1)
            train_acc = metric(y_pred_class, y_batch_class)

            # Update accumulated values
            total_correct += torch.sum(y_pred_class == y_batch_class).item()

            total_samples += y_batch.size(0) * y_batch.size(1)

            # Calculate accuracy for the current batch
            batch_acc = total_correct / total_samples

            # Update the progress bar description with the current accuracy
            epoch_progress.set_description(f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {metric.compute()*100:.4f}%", refresh=True)
            # total_correct = 0 
        # Print a newline to move to the next line after the epoch is finished
        print(f'Epoch {epoch + 1}/{epochs}, Train Accuracy: {metric.compute()*100:.4f}%')
        metric.reset()