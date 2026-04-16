import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from tqdm import tqdm
import math

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
MAX_VOCAB_SIZE = 20000
MAX_LEN        = 256
EMBED_DIM      = 256
HIDDEN_DIM     = 256
N_LAYERS       = 2
DROPOUT        = 0.3
BATCH_SIZE     = 64
N_EPOCHS       = 30
PATIENCE       = 5
N_HEADS        = 8
FF_DIM         = 1024
TF_N_LAYERS    = 3
TF_LR          = 3e-4
LABEL_SMOOTH   = 0.05


def preprocess_text(text):
    """Clean and tokenize text"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        return tokens
    return []


# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────
class Vocabulary:
    def __init__(self, max_size):
        self.max_size   = max_size
        self.word2idx   = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word   = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size       = 3

    def add_word(self, word):
        if word not in self.word_count:
            self.word_count[word] = 0
        self.word_count[word] += 1

    def build_vocab(self):
        sorted_words = sorted(self.word_count.items(),
                              key=lambda x: x[1], reverse=True)
        capacity = self.max_size - 3
        for word, _ in sorted_words[:capacity]:
            if word not in self.word2idx:
                idx = self.size
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
                self.size += 1

    def text_to_indices(self, tokens, max_len, model_type='lstm'):
        if model_type == 'transformer':
            # Prepend <cls>, skip unknowns
            indices = [self.word2idx["<cls>"]]
            for token in tokens:
                if token in self.word2idx:
                    indices.append(self.word2idx[token])
            indices = indices[:max_len]
        else:
            indices = [self.word2idx.get(t, self.word2idx["<unk>"])
                       for t in tokens]
            indices = indices[:max_len]

        pad_len = max_len - len(indices)
        indices = indices + [self.word2idx["<pad>"]] * pad_len
        return indices


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, dataframe, vocabulary, max_len,
                 is_training=True, model_type='lstm'):
        self.vocabulary  = vocabulary
        self.max_len     = max_len
        self.model_type  = model_type
        self.is_training = is_training

        self.texts  = []
        self.labels = []
        for _, row in dataframe.iterrows():
            tokens  = preprocess_text(row['text'])
            indices = vocabulary.text_to_indices(tokens, max_len, model_type)
            self.texts.append(indices)
            self.labels.append(int(row['label']))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.texts[idx], dtype=torch.long)
        label       = torch.tensor([self.labels[idx]], dtype=torch.float32)

        if self.model_type == 'transformer':
            # 1 for real tokens, 0 for padding
            attention_mask = (text_tensor != 0).long()
            return text_tensor, attention_mask, label
        else:
            return text_tensor, label


# ─────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────
class LSTM(nn.Module):
    def __init__(self, vocab_size=MAX_VOCAB_SIZE + 3, embed_dim=128,
                 hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                 dropout=DROPOUT, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        # Concat final forward & backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden)


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=MAX_LEN + 1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, E]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# Transformer Encoder
# Layer names must be: layer_norm, fc  (grader requirement)
# Architecture: CLS + mean-pool → sum → LayerNorm → Linear
# ─────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=MAX_VOCAB_SIZE + 3, embed_dim=EMBED_DIM,
                 n_heads=N_HEADS, n_layers=TF_N_LAYERS, ff_dim=FF_DIM,
                 dropout=DROPOUT, pad_idx=0):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim,
                                         padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True)           # Pre-LN for stable gradients
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # ── Head (names must match what grader expects) ───────────────────
        self.layer_norm = nn.LayerNorm(embed_dim)   # fuse CLS + mean → [B, E]
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(embed_dim, 1)   # project fused [B, E] → [B, 1]

    def forward(self, input_ids, attention_mask=None):
        embedded = self.pos_encoding(self.embedding(input_ids))  # [B, seq, E]

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        encoded = self.transformer_encoder(
            embedded, src_key_padding_mask=key_padding_mask)      # [B, seq, E]

        # ── CLS token ────────────────────────────────────────────────────
        cls_out = encoded[:, 0, :]                                 # [B, E]

        # ── Masked mean pool ─────────────────────────────────────────────
        if attention_mask is not None:
            mask_f   = attention_mask.unsqueeze(-1).float()        # [B, seq, 1]
            mean_out = (encoded * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            mean_out = encoded.mean(dim=1)                         # [B, E]

        # ── Fuse by addition → LayerNorm → classify ───────────────────────
        fused = self.layer_norm(cls_out + mean_out)                # [B, E]
        return self.fc(self.dropout(fused))                        # [B, 1]


# ─────────────────────────────────────────────
# Warmup + Cosine LR scheduler (step-level)
# ─────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = max(factor, self.min_lr / max(b for b in self.base_lrs))
        return [base_lr * factor for base_lr in self.base_lrs]


# ─────────────────────────────────────────────
# Label-smoothed BCE loss
# ─────────────────────────────────────────────
def smooth_bce(logits, targets, smoothing=LABEL_SMOOTH):
    targets_s = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, targets_s)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_and_preprocess_data(data_path, data_type='train',
                             model_type='lstm', shared_vocab=None):
    """
    Returns:
        data_type == 'train': (DataLoader, Vocabulary)
        otherwise           :  DataLoader
    """
    df = pd.read_parquet(data_path)

    if data_type == 'train':
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df['label'])

        vocab = Vocabulary(MAX_VOCAB_SIZE)
        for text in train_df['text']:
            for token in preprocess_text(text):
                vocab.add_word(token)
        vocab.build_vocab()

        train_dataset = IMDBDataset(train_df, vocab, MAX_LEN,
                                    is_training=True, model_type=model_type)
        train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0, pin_memory=True)
        return train_loader, vocab

    else:
        assert shared_vocab is not None, \
            "shared_vocab must be provided for non-train data"
        dataset = IMDBDataset(df, shared_vocab, MAX_LEN,
                              is_training=False, model_type=model_type)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)
        return loader


# ─────────────────────────────────────────────
# Training / evaluation loops
# ─────────────────────────────────────────────
def train(model, iterator, optimizer, criterion, device,
          model_type='lstm', scheduler=None):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(iterator, desc="Train", leave=False):
        optimizer.zero_grad()

        if model_type == 'transformer':
            text, attention_mask, labels = batch
            text, attention_mask, labels = (text.to(device),
                                            attention_mask.to(device),
                                            labels.to(device))
            predictions = model(text, attention_mask)
        else:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions  = model(text)

        loss = criterion(predictions, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        preds       = (torch.sigmoid(predictions) >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return epoch_loss / len(iterator), correct / total


def evaluate(model, iterator, criterion, device, model_type='lstm'):
    model.eval()
    epoch_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Eval ", leave=False):
            if model_type == 'transformer':
                text, attention_mask, labels = batch
                text, attention_mask, labels = (text.to(device),
                                                attention_mask.to(device),
                                                labels.to(device))
                predictions = model(text, attention_mask)
            else:
                text, labels = batch
                text, labels = text.to(device), labels.to(device)
                predictions  = model(text)

            loss        = criterion(predictions, labels)
            epoch_loss += loss.item()
            preds       = (torch.sigmoid(predictions) >= 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return epoch_loss / len(iterator), correct / total


def _build_val_loader(val_df, vocab, model_type):
    val_dataset = IMDBDataset(val_df, vocab, MAX_LEN,
                              is_training=False, model_type=model_type)
    return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DATA_PATH = "/kaggle/input/datasets/gaziaaziz1/dataset-1/hw5_train.parquet"

    full_df = pd.read_parquet(DATA_PATH)
    train_df, val_df = train_test_split(
        full_df, test_size=0.1, random_state=42, stratify=full_df['label'])

    # ═══════════════════════════════
    #  Train LSTM
    # ═══════════════════════════════
    # print("\n========== LSTM ==========")
    # lstm_train_loader, vocab = load_and_preprocess_data(
    #     DATA_PATH, data_type='train', model_type='lstm')
    # lstm_val_loader = _build_val_loader(val_df, vocab, 'lstm')

    # lstm_model = LSTM(
    #     vocab_size = MAX_VOCAB_SIZE + 3,
    #     embed_dim  = 128,
    #     hidden_dim = HIDDEN_DIM,
    #     n_layers   = N_LAYERS,
    #     dropout    = DROPOUT,
    #     pad_idx    = vocab.word2idx["<pad>"]
    # ).to(device)

    # lstm_criterion  = nn.BCEWithLogitsLoss()
    # lstm_optimizer  = optim.Adam(lstm_model.parameters(), lr=1e-3)
    # lstm_scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    #     lstm_optimizer, patience=2, factor=0.5)

    best_val_acc, patience_ctr = 0.0, 0
    # for epoch in range(1, N_EPOCHS + 1):
    #     tr_loss, tr_acc = train(lstm_model, lstm_train_loader,
    #                             lstm_optimizer, lstm_criterion, device, 'lstm')
    #     vl_loss, vl_acc = evaluate(lstm_model, lstm_val_loader,
    #                                lstm_criterion, device, 'lstm')
    #     lstm_scheduler.step(vl_loss)
    #     print(f"Epoch {epoch:02d} | Train {tr_acc:.4f} | Val {vl_acc:.4f}")
    #     if vl_acc > best_val_acc:
    #         best_val_acc = vl_acc
    #         torch.save(lstm_model.state_dict(), "lstm.pt")
    #         patience_ctr = 0
    #     else:
    #         patience_ctr += 1
    #         if patience_ctr >= PATIENCE:
    #             print("Early stopping LSTM.")
    #             break
    # print(f"Best LSTM Val Acc: {best_val_acc:.4f}")

    # ═══════════════════════════════
    #  Train Transformer
    # ═══════════════════════════════
    print("\n========== TransformerEncoder ==========")
    tf_train_loader, tf_vocab = load_and_preprocess_data(
        DATA_PATH, data_type='train', model_type='transformer')
    tf_val_loader = _build_val_loader(val_df, tf_vocab, 'transformer')

    tf_model = TransformerEncoder(
        vocab_size = MAX_VOCAB_SIZE + 3,
        embed_dim  = EMBED_DIM,
        n_heads    = N_HEADS,
        n_layers   = TF_N_LAYERS,
        ff_dim     = FF_DIM,
        dropout    = DROPOUT,
        pad_idx    = tf_vocab.word2idx["<pad>"],
    ).to(device)

    steps_per_epoch = len(tf_train_loader)
    total_steps     = N_EPOCHS * steps_per_epoch
    warmup_steps    = 2 * steps_per_epoch

    tf_optimizer = optim.AdamW(tf_model.parameters(),
                               lr=TF_LR, weight_decay=1e-2,
                               betas=(0.9, 0.98))
    tf_scheduler = WarmupCosineScheduler(tf_optimizer,
                                         warmup_steps=warmup_steps,
                                         total_steps=total_steps,
                                         min_lr=1e-6)

    best_val_acc, patience_ctr = 0.0, 0
    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = train(tf_model, tf_train_loader,
                                tf_optimizer, smooth_bce, device,
                                'transformer', scheduler=tf_scheduler)
        vl_loss, vl_acc = evaluate(tf_model, tf_val_loader,
                                   smooth_bce, device, 'transformer')
        lr_now = tf_optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | LR {lr_now:.2e} | "
              f"Train {tr_acc:.4f} | Val {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(tf_model.state_dict(), "transformer.pt")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("Early stopping Transformer.")
                break
    print(f"Best Transformer Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()