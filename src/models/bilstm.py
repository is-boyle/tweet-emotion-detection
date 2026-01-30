import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, num_classes=5, dropout=0.2, embedding_matrix=None, freeze_embeddings=False):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = torch.mean(lstm_out, dim=1)
        out = self.fc(out)
        return out

class TweetsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = text.split()[:self.max_len]
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def load_glove(path, embed_dim):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            if len(vector) == embed_dim:
                embeddings[word] = vector
    return embeddings

def build_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def build_embedding_matrix(vocab, glove, embed_dim):
    matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))

    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]

    return torch.tensor(matrix, dtype=torch.float)

def train_bilstm(df_train, df_test, df_validate, text_col='text', label_col='label', batch_size=32, epochs=5, lr=0.001, use_glove=True):
    vocab = build_vocab(df_train[text_col].values)

    train_dataset = TweetsDataset(df_train[text_col].values, df_train[label_col].values, vocab)
    test_dataset = TweetsDataset(df_test[text_col].values, df_test[label_col].values, vocab)
    #validate_dataset = TweetsDataset(df_validate[text_col].values, df_validate[label_col].values, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    #validate_loader = DataLoader(validate_dataset, batch_size=batch_size)

    embedding_matrix = None
    embed_dim = 100 if use_glove else 128
    if use_glove:
        glove = load_glove("data/glove.twitter.27B.100d.txt", embed_dim=embed_dim)
        embedding_matrix = build_embedding_matrix(vocab, glove, embed_dim=embed_dim)
    model = BiLSTMModel(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=len(np.unique(df_train[label_col].values)), embedding_matrix=embedding_matrix, freeze_embeddings=False)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train[label_col].values),
        y=df_train[label_col].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    model.eval()

    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(classification_report(all_labels, all_preds, digits=4))
    accuracy = correct / total
    return model, vocab, accuracy