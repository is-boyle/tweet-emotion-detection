import pandas as pd

def load_data(train_path=r"data/train.parquet", test_path=r"data/test.parquet", validation_path=r"data/validation.parquet"):
    return pd.read_parquet(train_path), pd.read_parquet(test_path), pd.read_parquet(validation_path)

