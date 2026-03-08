import re, pandas as pd
from sklearn.model_selection import train_test_split
from langdetect import detect

def clean_text(s):
    s = str(s)
    s = re.sub(r'http\S+','', s)
    s = re.sub(r'<.*?>','', s)
    s = re.sub(r'[^0-9A-Za-z\s\.,]', ' ', s)
    s = re.sub(r'\s+',' ', s).strip()
    return s.lower()

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['text_clean'] = df['text'].apply(clean_text)
    df['lang'] = df['text'].apply(lambda t: detect(t) if len(str(t).strip())>0 else 'unknown')
    return df

def stratified_split(df, strat_col='label', test_size=0.15, val_size=0.15, seed=42):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[strat_col], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val[strat_col], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
