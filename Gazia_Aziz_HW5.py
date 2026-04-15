import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string, re, math, os
from tqdm import tqdm

MAX_VOCAB_SIZE = 20000
MAX_LEN = 256
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.3
N_HEADS = 4
N_ENCODER_LAYERS = 2
FF_DIM = 512
N_EPOCHS = 10
LR = 1e-3

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        return word_tokenize(text)
    return []


class Vocabulary:
    def __init__(self, max_size):
        self.max_size = max_size
        self.word2idx = {"<pad>":0,"<unk>":1,"<cls>":2}
        self.idx2word = {0:"<pad>",1:"<unk>",2:"<cls>"}
        self.word_count = {}
        self.size = 3

    def add_word(self, word):
        self.word_count[word] = self.word_count.get(word,0)+1

    def build_vocab(self):
        sorted_words = sorted(self.word_count.items(), key=lambda x:x[1], reverse=True)
        for word,_ in sorted_words[:self.max_size-self.size]:
            self.word2idx[word] = self.size
            self.idx2word[self.size] = word
            self.size += 1

    def text_to_indices(self, tokens, max_len, model_type='lstm'):
        if model_type == 'transformer':
            indices = [self.word2idx["<cls>"]]
            for t in tokens:
                if t in self.word2idx:
                    indices.append(self.word2idx[t])
        else:
            indices = [self.word2idx.get(t,1) for t in tokens]

        indices = indices[:max_len]
        indices += [0]*(max_len-len(indices))
        return indices

class IMDBDataset(Dataset):
    def __init__(self, df, vocab, max_len, model_type='lstm'):
        self.labels = df['label'].tolist()
        self.indices = [
            vocab.text_to_indices(preprocess_text(t), max_len, model_type)
            for t in df['text']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx]), torch.tensor(self.labels[idx])

class LSTM(nn.Module):
    def __init__(self, vocab_size=MAX_VOCAB_SIZE+3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)

        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            num_layers=1,          
            batch_first=True,
            bidirectional=False    
        )

        self.fc = nn.Linear(HIDDEN_DIM, 2)  # ✅ fix

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]   # ✅ fix
        return self.fc(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(0)

    def forward(self,x):
        return x + self.pe[:,:x.size(1)].to(x.device)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=MAX_VOCAB_SIZE+3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.pos = PositionalEncoding(EMBEDDING_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_ENCODER_LAYERS)
        self.fc = nn.Linear(EMBEDDING_DIM, 2)

    def forward(self, x):
        mask = (x != 0)
        x = self.pos(self.embedding(x))
        out = self.encoder(x, src_key_padding_mask=~mask)
        return self.fc(out[:,0])


def load_and_preprocess_data(data_path, data_type='train', model_type='lstm', shared_vocab=None):
    df = pd.read_parquet(data_path)

    if data_type == 'train':
        vocab = Vocabulary(MAX_VOCAB_SIZE)
        for t in df['text']:
            for w in preprocess_text(t):
                vocab.add_word(w)
        vocab.build_vocab()

        train_df, val_df = train_test_split(df, test_size=0.1)

        train_loader = DataLoader(IMDBDataset(train_df, vocab, MAX_LEN, model_type),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(IMDBDataset(val_df, vocab, MAX_LEN, model_type),
                                  batch_size=BATCH_SIZE)

        train_loader.val_loader = val_loader
        return train_loader, vocab

    else:
        return DataLoader(IMDBDataset(df, shared_vocab, MAX_LEN, model_type),
                          batch_size=BATCH_SIZE)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("submission", exist_ok=True)

    train_loader, vocab = load_and_preprocess_data("hw5_train.parquet")
    val_loader = train_loader.val_loader

    model = LSTM().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    for _ in range(N_EPOCHS):
        train(model, train_loader, opt, crit, device)

    torch.save(model.state_dict(),"submission/lstm.pt")

    train_loader, vocab = load_and_preprocess_data("hw5_train.parquet","train","transformer")

    model = TransformerEncoder().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    for _ in range(N_EPOCHS):
        train(model, train_loader, opt, crit, device)

    torch.save(model.state_dict(),"submission/transformer.pt")

if __name__ == "__main__":
    main()