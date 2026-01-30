import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from data_loader import load_data
from models.baseline import load_model as load_baseline
from itertools import combinations
from models.bilstm import TweetsDataset as BiLSTMDataset, BiLSTMModel, load_glove, build_embedding_matrix
from models.bert import TweetsDataset as BertDataset
from models.distilbert import TweetsDataset as DistilBertDataset
from models.roberta import TweetsDataset as RobertaDataset
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score


def paired_bootstrap(y_true, y_pred_a, y_pred_b, n_resamples=10000):

    n = len(y_true)
    diffs = []

    for _ in range(n_resamples):
        idx = np.random.choice(n, n, replace=True)
        f1_a = f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    low, high = np.percentile(diffs, [2.5, 97.5])

    return np.mean(diffs), low, high


def predict_baseline(df):
    model = load_baseline()
    preds = model.predict(df['text'].values)
    return np.array(preds)


def predict_bilstm(df, vocab, use_glove=False):
    embedding_matrix = None
    file_name = 'bilstm_glove.pth' if use_glove else 'bilstm.pth'
    embed_dim = 100 if use_glove else 128
    if use_glove:
        glove = load_glove("data/glove.twitter.27B.100d.txt", embed_dim=100)
        embedding_matrix = build_embedding_matrix(vocab, glove, embed_dim=100)

    model = BiLSTMModel(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=6, embedding_matrix=embedding_matrix, freeze_embeddings=False)
    model.load_state_dict(torch.load(f'models/{file_name}'))
    model.eval()

    dataset = BiLSTMDataset(df["text"].values, [0]*len(df), vocab)
    loader = DataLoader(dataset, batch_size=32)

    preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.tolist())

    return np.array(preds)


def predict_bert(df, tokenizer):
    model = BertForSequenceClassification.from_pretrained('models/bert')
    model.eval()
    device = next(model.parameters()).device

    dataset = BertDataset(df['text'].values, [0]*len(df), tokenizer)  # dummy labels
    loader = DataLoader(dataset, batch_size=32)

    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            preds.extend(predicted.tolist())

    return np.array(preds)


def predict_distilbert(df, tokenizer):
    model = DistilBertForSequenceClassification.from_pretrained('models/distilbert')
    model.eval()
    device = next(model.parameters()).device

    dataset = DistilBertDataset(df['text'].values, [0]*len(df), tokenizer)  # dummy labels
    loader = DataLoader(dataset, batch_size=32)

    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            preds.extend(predicted.tolist())

    return np.array(preds)


def predict_roberta(df, tokenizer):
    model = RobertaForSequenceClassification.from_pretrained('models/roberta')
    model.eval()
    device = next(model.parameters()).device

    dataset = RobertaDataset(df['text'].values, [0]*len(df), tokenizer)  # dummy labels
    loader = DataLoader(dataset, batch_size=32)

    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            preds.extend(predicted.tolist())

    return np.array(preds)


def main():
    _, df_test, _ = load_data()

    preds_dict = {}
    y_true = df_test['label'].values

    # Baseline
    try:
        preds = predict_baseline(df_test)
        preds_dict['baseline'] = preds
        results = classification_report(y_true, preds, digits=4)
        print("=== Baseline Evaluation Results ===")
        print(results)
        with open('results/baseline_evaluation.txt', 'w') as f:
            f.write(results)
        print()
    except Exception as e:
        print(e)

    # BiLSTM
    try:
        vocab = torch.load('models/bilstm_vocab.pth')
        preds = predict_bilstm(df_test, vocab, use_glove=False)
        preds_dict['bilstm'] = preds
        results = classification_report(y_true, preds, digits=4)
        print("=== BiLSTM Evaluation Results ===")
        print(results)
        with open('results/bilstm_evaluation.txt', 'w') as f:
            f.write(results)
        print()
    except Exception as e:
        print(e)

    # BiLSTM with GloVe
    try:
        vocab = torch.load('models/bilstm_vocab.pth')
        preds = predict_bilstm(df_test, vocab, use_glove=True)
        preds_dict['bilstm_glove'] = preds
        results = classification_report(y_true, preds, digits=4)
        print("=== BiLSTM with GloVe Evaluation Results ===")
        print(results)
        with open('results/bilstm_glove_evaluation.txt', 'w') as f:
            f.write(results)
        print()
    except Exception as e:
        print(e)

    # BERT
    try:
        tokenizer = BertTokenizer.from_pretrained('models/bert')
        preds = predict_bert(df_test, tokenizer)
        preds_dict['bert'] = preds
        results = classification_report(y_true, preds, digits=4)
        print("=== BERT Evaluation Results ===")
        print(results)
        with open('results/bert_evaluation.txt', 'w') as f:
            f.write(results)
        print()
    except Exception as e:
        print(e)

    # DistilBERT
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert')
        preds = predict_distilbert(df_test, tokenizer)
        preds_dict['distilbert'] = preds
        results = classification_report(y_true, preds, digits=4)
        print("=== DistilBERT Evaluation Results ===")
        print(results)
        with open('results/distilbert_evaluation.txt', 'w') as f:
            f.write(results)
        print()
    except Exception as e:
        print(e)

    # RoBERTa
    try:
        tokenizer = RobertaTokenizer.from_pretrained('models/roberta')
        preds = predict_roberta(df_test, tokenizer)
        preds_dict['roberta'] = preds
        results = classification_report(y_true, preds, digits=4)
        print("=== RoBERTa Evaluation Results ===")
        print(results)
        with open('results/roberta_evaluation.txt', 'w') as f:
            f.write(results)
        print()
    except Exception as e:
        print(e)

    # Paired bootstrap comparisons
    try:
        pairs = list(combinations(list(preds_dict.keys()), 2))
        lines = []
        lines.append("Paired bootstrap comparisons (mean F1 diff, 95% CI):\n")
        for a, b in pairs:
            mean_diff, low, high = paired_bootstrap(y_true, preds_dict[a], preds_dict[b])
            signif = "(significant)" if not (low <= 0 <= high) else "(not significant)"
            line = f"{a} vs {b}: mean_diff={mean_diff:.4f}, 95% CI=[{low:.4f}, {high:.4f}] {signif}"
            print(line)
            lines.append(line)

        with open('results/paired_bootstrap_results.txt', 'w') as f:
            f.write("\n".join(lines))

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()