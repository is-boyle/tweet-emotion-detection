from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class TweetsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds), 
        'f1_macro': f1_score(labels, preds, average='macro')
    }

def train_roberta(df_train, df_test, df_validate, text_col='text', label_col='label', epochs=3, batch_size=16, lr=2e-5):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(np.unique(df_train[label_col].values, return_counts=False)))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train[label_col].values),
        y=df_train[label_col].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_dataset = TweetsDataset(df_train[text_col].values, df_train[label_col].values, tokenizer)
    test_dataset = TweetsDataset(df_test[text_col].values, df_test[label_col].values, tokenizer)
    validate_dataset = TweetsDataset(df_validate[text_col].values, df_validate[label_col].values, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=42
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validate_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    results = trainer.evaluate(eval_dataset=test_dataset)
    accuracy = results['eval_accuracy']
    f1 = results["eval_f1_macro"]
    print(f'BERT Test Accuracy: {accuracy:.4f}')
    print(f"BERT Test Macro F1: {f1:.4f}")

    return model, tokenizer, accuracy
