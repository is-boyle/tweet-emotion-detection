import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data
import argparse
import torch
from models.baseline import train as train_baseline, save_model
from models.bilstm import train_bilstm
from models.bert import train_bert
from models.distilbert import train_distilbert
from models.roberta import train_roberta


def main(model_type):
    df_train, df_test, df_validation = load_data()
    os.makedirs('models', exist_ok=True)

    if model_type == 'baseline':
        model, report = train_baseline(df_train, df_test, df_validation)
        save_model(model, 'models/baseline.joblib')
        print('Baseline training complete. Classification report:')
        print(report)
    elif model_type == 'bilstm':
        model, vocab, acc = train_bilstm(df_train, df_test, df_validation, use_glove=False)
        torch.save(model.state_dict(), 'models/bilstm.pth')
        torch.save(vocab, 'models/bilstm_vocab.pth')
        print(f'BiLSTM training complete. Accuracy: {acc:.4f}')
    elif model_type == 'bilstm-glove':
        model, vocab, acc = train_bilstm(df_train, df_test, df_validation, use_glove=True)
        torch.save(model.state_dict(), 'models/bilstm_glove.pth')
        torch.save(vocab, 'models/bilstm_vocab.pth')
        print(f'BiLSTM with GloVe training complete. Accuracy: {acc:.4f}')
    elif model_type == 'bert':
        model, tokenizer, acc = train_bert(df_train, df_test, df_validation)
        model.save_pretrained('models/bert')
        tokenizer.save_pretrained('models/bert')
        print(f'BERT training complete. Accuracy: {acc:.4f}')
    elif model_type == 'distilbert':
        model, tokenizer, acc = train_distilbert(df_train, df_test, df_validation)
        model.save_pretrained('models/distilbert')
        tokenizer.save_pretrained('models/distilbert')
        print(f'DistilBERT training complete. Accuracy: {acc:.4f}')
    elif model_type == 'roberta':
        model, tokenizer, acc = train_roberta(df_train, df_test, df_validation)
        model.save_pretrained('models/roberta')
        tokenizer.save_pretrained('models/roberta')
        print(f'RoBERTa training complete. Accuracy: {acc:.4f}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'bilstm', 'bilstm-glove', 'bert', 'distilbert', 'roberta'])
    args = parser.parse_args()
    main(args.model)
