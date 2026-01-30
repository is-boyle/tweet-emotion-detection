from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

def train(df_train, df_test, df_validate, text_col='text', label_col='label'):
    X_train = df_train[text_col].values
    X_test = df_test[text_col].values
    X_validate = df_validate[text_col].values
    y_train = df_train[label_col].values
    y_test = df_test[label_col].values
    y_validate = df_validate[label_col].values
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    report = classification_report(y_test, preds)
    return pipe, report

def save_model(pipe, path=r'models/baseline.joblib'):
    joblib.dump(pipe, path)


def load_model(path=r'models/baseline.joblib'):
    return joblib.load(path)