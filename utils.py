# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch.nn.functional as F
from nltk.corpus import wordnet

import os
import random

from collections import Counter

import zipfile
import requests
import seaborn as sns
import re
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import ngrams
import string

import nltk
# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pred):
    """
    Computes evaluation metrics during the training process.
    Expects pred to have attributes: label_ids and predictions
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')  # weighted for multi-class
    }

# -----------------------------
# Plotting
# -----------------------------
def plot_training_history(trainer):
    """
    Visualizes the training and validation loss curves.
    Expects trainer.state.log_history as a list of dicts with 'loss' and 'eval_loss'.
    """
    history = trainer.state.log_history
    df_history = pd.DataFrame(history)

    # Filter for training loss and validation loss entries
    train_loss = df_history[df_history['loss'].notna()][['epoch', 'loss']]
    eval_loss = df_history[df_history['eval_loss'].notna()][['epoch', 'eval_loss']]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss', color='red', marker='o', markersize=3)
    plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', color='blue', marker='x', markersize=3)

    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('images/training_loss_curve.png', dpi=300)
    plt.show()

def plot_metrics(trainer):
    """
    Visualizes Accuracy and F1-Score over training epochs.
    Expects trainer.state.log_history with 'eval_accuracy' and 'eval_f1'.
    """
    df_history = pd.DataFrame(trainer.state.log_history)
    eval_metrics = df_history[df_history['eval_accuracy'].notna()]

    plt.figure(figsize=(10, 6))
    plt.plot(eval_metrics['epoch'], eval_metrics['eval_accuracy'], label='Accuracy', marker='o')
    plt.plot(eval_metrics['epoch'], eval_metrics['eval_f1'], label='F1-Score', marker='s')

    plt.title('Validation Metrics per Epoch', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('images/validation_metrics.png', dpi=300)
    plt.show()

# -----------------------------
# Data cleaning
# -----------------------------
def clean_duplicates(df: pd.DataFrame, title_col: str = 'title', text_col: str = 'text') -> pd.DataFrame:
    """
    Remove exact duplicates, duplicate/empty titles, and duplicate/empty text.
    """
    # 1. Exact duplicates
    total_exact_duplicates = df.duplicated().sum()
    print(f"Total exact duplicate rows (all columns): {total_exact_duplicates}")
    if total_exact_duplicates > 0:
        print("Example exact duplicates:")
        print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(4))
    df = df.drop_duplicates()
    print(f"Rows after removing exact duplicates: {len(df)}\n")

    # 2. Duplicate or empty titles
    df = df[~df[title_col].isna() & (df[title_col].str.strip() != "")]
    print(f"Rows after removing empty titles: {len(df)}")
    duplicate_title_mask = df[title_col].duplicated(keep=False)
    num_duplicate_titles = duplicate_title_mask.sum()
    print(f"Total rows with duplicate titles: {num_duplicate_titles}")
    if num_duplicate_titles > 0:
        print("Example duplicate titles:")
        print(df[duplicate_title_mask].sort_values(title_col).head(4))
    df = df.drop_duplicates(subset=title_col, keep='first')
    print(f"Total rows after removing duplicate titles: {len(df)}\n")

    # 3. Duplicate or empty text
    df = df[~df[text_col].isna() & (df[text_col].str.strip() != "")]
    print(f"Rows after removing empty text: {len(df)}")
    duplicate_text_mask = df[text_col].duplicated(keep=False)
    num_duplicated_text_rows = duplicate_text_mask.sum()
    print(f"Total rows with duplicated text: {num_duplicated_text_rows}")
    if num_duplicated_text_rows > 0:
        print("Example duplicated text rows:")
        print(df[duplicate_text_mask].sort_values(text_col).head(4))
    df = df.drop_duplicates(subset=text_col, keep='first')
    print(f"Total rows after removing duplicated text: {len(df)}\n")

    return df

def clean_dates(df: pd.DataFrame, date_col: str = 'date', drop_invalid: bool = True) -> pd.DataFrame:
    """
    Clean and standardize a date column, fixing month abbreviations and converting to datetime.
    """
    month_map = {
        'Sept': 'Sep',
        'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
        'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
        'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
    }

    def clean_date_string(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        s = re.sub(r'\s+', ' ', s)
        for k, v in month_map.items():
            s = re.sub(r'\b'+k+r'\b', v, s, flags=re.IGNORECASE)
        return s

    df[date_col] = df[date_col].apply(clean_date_string)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)

    num_invalid = df[date_col].isna().sum()
    if drop_invalid:
        print(f"Dropping {num_invalid} rows with invalid dates.")
        df = df.dropna(subset=[date_col]).reset_index(drop=True)
    else:
        print(f"{num_invalid} rows have invalid dates (kept as NaT).")

    return df

def get_top_ngrams(corpus, n=2, top_k=10):
    # Convert all items to str and skip NaNs
    corpus = [str(x) for x in corpus if pd.notna(x)]
    words = ' '.join(corpus).lower().split()
    grams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(grams).most_common(top_k)


def clean_for_classical_ml(text):
    """
    Aggressive cleaning for TF-IDF based models.
    Removes stop words, performs lemmatization and removes all noise.
    """
    text = str(text).lower()
    text = re.sub(r'^.*?(reuters|21st century wire|image via)\s*[-—]\s*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(cleaned)

def clean_for_bert(text):
    """
    Minimal cleaning for Transformer models.
    Keeps stop words and punctuation for context, but removes obvious leakage.
    """
    text = str(text)
    text = re.sub(r'^.*?(reuters|REUTERS|21st Century Wire|IMAGE VIA)\s*[-—]\s*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bigram_freq(corpus):
    corpus = [str(x) for x in corpus if pd.notna(x)]
    words = ' '.join(corpus).lower().split()
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    return Counter(bigrams)

def create_comparison_clouds(fake_f, real_f, folder_name="images", type="classicML"):
    """Generates WordClouds and saves the result to a specified folder."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")

    wc_params = {
        "width": 1200,
        "height": 800,
        "max_words": 100,
        "background_color": "white",
        "collocations": False
    }

    cloud_fake = WordCloud(**wc_params, colormap='Reds').generate_from_frequencies(fake_f)
    cloud_real = WordCloud(**wc_params, colormap='Blues').generate_from_frequencies(real_f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(cloud_fake, interpolation='bilinear')
    ax1.set_title('Top Bigrams: Fake News', fontsize=26, pad=20, color='darkred')
    ax1.axis('off')

    ax2.imshow(cloud_real, interpolation='bilinear')
    ax2.set_title('Top Bigrams: Real News', fontsize=26, pad=20, color='darkblue')
    ax2.axis('off')

    plt.tight_layout(pad=5)

    # Save the combined plot to the folder
    save_path = os.path.join(folder_name, f"bigram_comparison_{type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Visualization saved to: {save_path}")
    plt.show()


def download_and_extract_liar():
    zip_path = "liar_dataset.zip"
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    if not os.path.exists(zip_path):
        print("Downloading LIAR dataset...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)

    # Unzip only if the target file (test.tsv) doesn't exist
    if not os.path.exists("test.tsv"):
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

    print("LIAR dataset is ready.")


def tokenize_single(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    return SingleTextDataset(enc)


def add_typos(text, rate=0.05):
    """
    Apply random character-level perturbations:
    - swap adjacent characters
    - delete a character
    - insert a random character
    """
    if not isinstance(text, str):
        return text

    chars = list(text)
    i = 0

    while i < len(chars):
        if random.random() < rate and chars[i].isalpha():
            typo_type = random.choice(["swap", "delete", "insert"])

            if typo_type == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 1

            elif typo_type == "delete":
                chars.pop(i)
                continue

            elif typo_type == "insert":
                chars.insert(i, random.choice(string.ascii_lowercase))
                i += 1
        i += 1

    return "".join(chars)


def inject_fake_source(text):
    """Inject fake Reuters attribution at the beginning"""
    return f"REUTERS - {text}"

def predict_label(text, tokenizer, model, device):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    return logits.argmax(dim=-1).item()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    return None


def clean_synonyms(word, pos):
    """Get usable synonyms only"""
    synsets = wordnet.synsets(word, pos=pos)
    candidates = set()

    for s in synsets:
        for l in s.lemmas():
            name = l.name().replace("_", " ")
            if (
                name.lower() != word.lower()
                and name.isalpha()
                and len(name) > 3
            ):
                candidates.add(name)

    return list(candidates)


def synonym_attack(text, rate=0.25, max_changes=15):
    """
    Targeted synonym substitution attack.
    Focuses on content words only.
    """
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    new_tokens = tokens.copy()
    changes = 0

    for i, (word, tag) in enumerate(pos_tags):
        if changes >= max_changes:
            break


def predict_label_with_confidence(text, tokenizer, model, device):
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)

    label = probs.argmax(dim=-1).item()
    confidence = probs.max(dim=-1).values.item()

    return label, confidence

