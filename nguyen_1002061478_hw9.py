import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class FakeNewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, h_n = self.rnn(self.embedding(x))
        return self.sigmoid(self.fc(h_n.squeeze(0)))


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, (h_n, _) = self.lstm(self.embedding(x))
        return self.sigmoid(self.fc(h_n.squeeze(0)))


def run_experiment(model, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(2): 
        for texts, labels in train_loader:
            optimizer.zero_grad(); outputs = model(texts).squeeze()
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts).squeeze(); predicted = (outputs > 0.5).float()
            total += labels.size(0); correct += (predicted == labels).sum().item()
    return (correct / total) * 100

if __name__ == "__main__":
    try:
        
        df_true = pd.read_csv('True.csv'); df_true['label'] = 0
        df_fake = pd.read_csv('Fake.csv'); df_fake['label'] = 1
        df = pd.concat([df_true, df_fake]).sample(frac=0.2).reset_index(drop=True) # Using 20% for speed

        
        all_text = " ".join(df['text'].astype(str)).lower().split()
        vocab = {word: i+1 for i, word in enumerate(set(all_text[:10000]))}
        vocab_size = len(vocab) + 1
        def encode(text): return [vocab.get(w.lower(), 0) for w in str(text).split()][:50]
        
        X = [encode(t) for t in df['text']]
        X = [seq + [0]*(50-len(seq)) for seq in X]
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        train_loader = DataLoader(FakeNewsDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(FakeNewsDataset(X_test, y_test), batch_size=32)

        
        rnn_acc = run_experiment(VanillaRNN(vocab_size, 32, 64), train_loader, test_loader)
        lstm_acc = run_experiment(LSTMModel(vocab_size, 32, 64), train_loader, test_loader)

        print(f"\nRNN Accuracy: {rnn_acc:.2f}% | LSTM Accuracy: {lstm_acc:.2f}%")
    except Exception as e: print(f"Error: {e}")