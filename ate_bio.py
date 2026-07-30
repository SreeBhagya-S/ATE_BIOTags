# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer
import json
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup

import nltk, string, re, spacy,unicodedata, random
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import ToktokTokenizer

#  Modify the Model for Classification
import torch.nn as nn

# class TransformerEncoderLayerWithAttn(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = nn.ReLU()


#     def forward(self, src, attention_mask=None):
#         key_padding_mask = None
#         if attention_mask is not None:
#             key_padding_mask = attention_mask == 0  # mask where attention_mask is 0 (padding)

#         attn_output, attn_weights = self.self_attn(
#             src, src, src, key_padding_mask=key_padding_mask, need_weights=True
#         )
#         src2 = self.norm1(src + self.dropout1(attn_output))
#         ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src2 = self.norm2(src2 + self.dropout2(ff_output))
#         return src2, attn_weights



# class TransformerEncoderModel(nn.Module):
#     def __init__(self, config):
#         super(TransformerEncoderModel, self).__init__()
#         self.embedding = nn.Embedding(config.vocab_size, config.d_model)
#         self.transformer_layers = nn.ModuleList([
#         TransformerEncoderLayerWithAttn(config.d_model, config.nhead, config.dim_feedforward, dropout=0.2)
#         for _ in range(config.num_layers)
#         ])
#         self.fc = nn.Linear(config.d_model, config.vocab_size)  # MLM Head
#         self.nsp_fc = nn.Linear(config.d_model, 2)  # NSP Head
#         self.dropout = nn.Dropout(0.1)
#         self.max_seq_length = config.max_seq_length
#         self.output_hidden_states = config.output_hidden_states  
#         self.absa_fc = nn.Linear(config.d_model, config.num_absa_classes)



#     def forward(self, src, attention_mask=None, return_hidden_states=None, return_attentions=False):
#         if return_hidden_states is None:
#             return_hidden_states = self.output_hidden_states

#         src = self.embedding(src) * (self.max_seq_length ** 0.5)
#         src = self.dropout(src)

#         attention_weights = []
#         hidden_states = []

#         x = src
#         for layer in self.transformer_layers:
#             x, attn = layer(x, attention_mask=attention_mask)
#             if return_hidden_states:
#                 hidden_states.append(x)
#             if return_attentions:
#                 attention_weights.append(attn)

#         output = x  # Final output from last layer
#         mlm_output = self.fc(output)
#         cls_token = output[:, 0, :]  # CLS for NSP
#         nsp_output = self.nsp_fc(cls_token)

#         results = [mlm_output, nsp_output]
#         if return_hidden_states:
#             results.append(output)
#         if return_attentions:
#             results.append(attention_weights)

#         absa_logits = self.absa_fc(output[:, 0, :])  # [CLS] token
#         results.append(absa_logits)

#         return tuple(results)


#     def extract_embeddings(self, src, output_hidden_states=False):
#         """Returns word embeddings, contextualized embeddings, and sentence embeddings."""
#         with torch.no_grad():
#             word_embeddings = self.embedding(src)
#             word_embeddings = self.dropout(word_embeddings)

#             transformer_input = word_embeddings.permute(1, 0, 2)
#             hidden_states = []
#             x = transformer_input
#             for layer in self.transformer_encoder.layers:
#                 x = layer(x)
#                 if output_hidden_states:
#                     hidden_states.append(x.permute(1, 0, 2))
#             x = x.permute(1, 0, 2)
#             sentence_embeddings = x[:, 0, :]

#         if output_hidden_states:
#             return word_embeddings, x, sentence_embeddings, hidden_states
#         return word_embeddings, x, sentence_embeddings

# class TransformerConfig:
#     def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, output_hidden_states=True, num_absa_classes=3):
#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.nhead = nhead
#         self.num_layers = num_layers
#         self.dim_feedforward = dim_feedforward
#         self.max_seq_length = max_seq_length
#         self.output_hidden_states = output_hidden_states
#         self.num_absa_classes = num_absa_classes 

#     @classmethod
#     def from_pretrained(cls, config_path):
#         with open(config_path, "r") as f:
#             config_dict = json.load(f)
#         return cls(**config_dict)

# config_path = "/content/drive/MyDrive/PHD_Corpus/7Jan_model_V2_config_14l_8BS_512SQ_100E_LR1e-4_TS25_RS5/7Jan_config_V2_14l_8BS_512SQ_100E_LR1e-4_TS25_RS5_12AH.json"
# config = TransformerConfig.from_pretrained(config_path)
# config.num_absa_classes = 5  # or 5, based on your task

# model = TransformerEncoderModel(config)


# print(model)


# from transformers import PreTrainedTokenizerFast

# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/content/drive/MyDrive/PHD_Corpus/tokenizer.json")
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # Load config and model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device,"alloted..............................................")
# model = TransformerEncoderModel(config).to(device)

# # Load MLM checkpoint (trained on MLM+NSP)
# checkpoint_path = "/content/drive/MyDrive/PHD_Corpus/7Jan_checkpoint.pth"
# checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# model.eval()

from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

from torch.utils.data import Dataset

# class AspectSentimentDataset(Dataset):
#     def __init__(self, df, tokenizer, label2id, max_len=128):
#         self.tokenizer = tokenizer
#         self.texts = df['Review_Text']
#         self.aspects = df['Aspect_Term']
#         self.categories = df['Aspect_Category']
#         self.labels = df['Sentiment_Class'].map(label2id)
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = f"{self.texts.iloc[idx]} [ASP] {self.aspects.iloc[idx]} [CAT] {self.categories.iloc[idx]}"
#         encoded = self.tokenizer(
#             text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
#         )
#         return (
#             encoded['input_ids'].squeeze(),
#             encoded['attention_mask'].squeeze(),
#             torch.tensor(self.labels.iloc[idx])
#         )

# from torch.utils.data import DataLoader, random_split

# dataset = AspectSentimentDataset(df, tokenizer, label2id, max_len=128)


"""**BIO Tagging**"""

!pip install --upgrade transformers

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

# Step 1: Load Data
df = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/Balanced_CM.csv")  # Make sure this has a 'Review_Text' column
texts = df['Review_Text'].tolist()

# Step 2: Load Multilingual BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# Step 3: Extract BERT Embeddings
def get_bert_embeddings(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_length - 2]  # Ensure within limit
    input_ids = tokenizer.encode(tokens, return_tensors='pt', truncation=True, max_length=max_length, add_special_tokens=True)
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state.squeeze(0)
    return tokens, embeddings


# Step 4: Collect All Candidate Tokens and Embeddings
all_tokens = []
all_embeddings = []

for review in texts:
    tokens, embeddings = get_bert_embeddings(review)
    for tok, emb in zip(tokens, embeddings):
        if tok.startswith('##'): continue  # Skip subwords
        all_tokens.append(tok)
        all_embeddings.append(emb.numpy())

print(f"Collected {len(all_tokens)} tokens for clustering...")

"""**Preprocessing and Tokenization**"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
torch.cuda.empty_cache()

!pip install emoji

# -------------------------
# Emoji Removal
# -------------------------
def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def clean_text(text):
    # Remove stray punctuation/symbols (keep Malayalam, English, digits, whitespace)
    text = re.sub(r"[^\w\s\u0D00-\u0D7F\u0D80-\u0DFF]", "", text)
    return text

# Convert to lowercase
texts = [text.lower() for text in texts]
texts = [remove_emojis(text) for text in texts]
#Check result
print(texts[:5])

import string
import re
import emoji
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import difflib
import unicodedata

lemmatizer_english = WordNetLemmatizer()

def lemmatize_en(word):
    return lemmatizer_english.lemmatize(word)

def load_suffixes(file_paths):
    suffixes = set()
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                suffix = line.strip()
                if suffix:
                    suffixes.add(suffix)
    return sorted(suffixes, key=len, reverse=True)  # longer suffixes first
mal_suffix_files = [
    "/content/drive/MyDrive/PHD_Corpus/malayalam_suffix_1.txt",
    "/content/drive/MyDrive/PHD_Corpus/malayalam_suffixmorph.txt"
]
MAL_SUFFIXES = load_suffixes(mal_suffix_files)


def generate_variants(word):
    variants = set()
    variants.add(word)
    variants.add(word + "ം")
    if word.endswith("ൻ്റെ") or word.endswith("ന്റെ"):
        variants.add(word[:-2])
    if word.endswith("ിൻ്റെ"):
        variants.add(word[:-3])
    if word.endswith("ന്റെ"):
        variants.add(word[:-3])
    variants.add(word.rstrip("ംസ്‌"))  # remove "ം" or trailing "സ്‌"
    return list(variants)

def normalize_malayalam(word):
    word = word.replace('\u200c', '')  # zero-width non-joiner
    word = word.replace('\u200d', '')  # zero-width joiner
    word = unicodedata.normalize('NFC', word)  # canonical form
    return word

def lemmatize_ml(word):
    word = normalize_malayalam(word)
    original = word
    suffixes = ["യുടെ", "ന്റെ", "നുള്ള", "ത്തിൽ", "കളെ", "ങ്ങൾ", "ങ്ങളായി", "ം", "മായി", "പ്പെട്ടു", "നിന്റെ"]
    suffixes.extend(MAL_SUFFIXES)

    for suffix in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            stem = normalize_malayalam(stem)

            if len(stem) >= 3:
                variants = generate_variants(stem)
                return stem

    # Try match full word directly too
    variants = generate_variants(word)
    return word

def lemmatize(word):
    if all('\u0D00' <= ch <= '\u0D7F' for ch in word):  # Malayalam
        return lemmatize_ml(word)
    else:
        return lemmatize_en(word)

def extract_compound_nouns(text):
    # pattern like സ്‌ക്രീനിന്റെ → സ്‌ക്രീൻ
    compound_nouns = []
    possessive_pattern = re.findall(r'(\S+ന്റെ)', text)
    for match in possessive_pattern:
        root = lemmatize_ml(match)
        if len(root) >= 4:
            compound_nouns.append(root)
    return compound_nouns



def merge_subwords(tokens):
    words = []
    current_word = ''
    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
            current_word = token
    if current_word:
        words.append(current_word)
    return words

def is_subword(token):
    return token.startswith("##")

def clean_token_fn(token):
    return token.replace("##", "").lower()

def load_stopwords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='windows-1252') as f:
            return set(line.strip() for line in f if line.strip())

# Load both files
eng_stopwords = load_stopwords('/content/drive/MyDrive/PHD_Corpus/englishST.txt')
mal_stopwords = load_stopwords('/content/drive/MyDrive/PHD_Corpus/mal_stopwords.txt')

# Merge and remove duplicates
all_stopwords = eng_stopwords.union(mal_stopwords)

# Save to new file
with open('/content/merged_stopwords.txt', 'w', encoding='utf-8') as f:
    for word in sorted(all_stopwords):
        f.write(word + '\n')

def load_sentiment_words_from_files(file_paths):
    sentiment_words = set()
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                words = {line.strip().lower() for line in f if line.strip()}
        except UnicodeDecodeError:
            with open(path, 'r', encoding='windows-1252') as f:
                words = {line.strip().lower() for line in f if line.strip()}
        sentiment_words.update(words)
    return sentiment_words

# Example usage with file paths
file_paths = [
    '/content/drive/MyDrive/PHD_Corpus/neg_word_malayalam.txt',
    '/content/drive/MyDrive/PHD_Corpus/pos_words_malayalam.txt',
    '//content/drive/MyDrive/PHD_Corpus/negative-words_english.txt',
    '/content/drive/MyDrive/PHD_Corpus/positive-words_english.txt'
]

SENTIMENT_WORDS = load_sentiment_words_from_files(file_paths)

"""**Named Entity Recognition (NER)& Semantic and Lexical Filtering**"""

def load_aspect_dict_and_keywords(filepath):
    aspect_dict = {}
    aspect_keywords = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        current_aspect = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith('-'):
                current_aspect = normalize_malayalam(line[:-1])
                aspect_dict[current_aspect] = []
            elif current_aspect:
                sub = normalize_malayalam(line)
                aspect_dict[current_aspect].append(sub)
                aspect_keywords.add(sub)

    return aspect_dict, aspect_keywords

# Load once at the top
ASPECT_DICT, ASPECT_KEYWORDS = load_aspect_dict_and_keywords("/content/drive/MyDrive/PHD_Corpus/aspect_terms.txt")

# Clean aspect keywords
ASPECT_KEYWORDS = set(clean_text(k) for k in ASPECT_KEYWORDS)

# Clean aspect dict keys and values
cleaned_dict = {}
for aspect, subterms in ASPECT_DICT.items():
    clean_aspect = clean_text(aspect)
    clean_subterms = [clean_text(s) for s in subterms]
    cleaned_dict[clean_aspect] = clean_subterms

ASPECT_DICT = cleaned_dict

ASPECT_DICT.items()

import torch.nn.functional as F

import torch

import string

STOPWORDS = load_stopwords('/content/merged_stopwords.txt')
PUNCTUATIONS = set(string.punctuation)

# Load mBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
SPECIAL_TOKEN_IDS = set(tokenizer.convert_tokens_to_ids(token) for token in tokenizer.all_special_tokens)

SPECIAL_TOKENS = set(tokenizer.all_special_tokens)


def process_embeddings(hidden_states, input_ids):
    last_hidden = hidden_states[-1]

    embeddings = last_hidden.squeeze(0)  # remove batch dimension; shape: (seq_len, hidden_dim)
    token_ids = input_ids.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    aspect_scores = torch.mean(embeddings, dim=1)

    filtered = []
    for i, (token, score) in enumerate(zip(tokens, aspect_scores)):
        token_id = token_ids[i].item()
        if token_id in SPECIAL_TOKEN_IDS:
            continue
        if token.lower() in all_stopwords:
            continue
        if token in PUNCTUATIONS or token.strip() == "":
            continue
        if any(special in token for special in ["<", ">", "[", "]", "▁"]):
            continue
        filtered.append((i, score.item()))

    top_k = min(50, len(filtered))
    top_indices = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
    return [i for i, _ in top_indices]


def extract_aspect_terms(text):
    text = remove_emojis(text)
    text = clean_text(text)

    compound_nouns = extract_compound_nouns(text)
    phrases = set()
    merged_tokens = []
    current_word = ""
    cleaned_tokens = []
    matched_aspects = set()
    aspect_hits = {}

    # Step 1: Match compound nouns directly
    for cn in compound_nouns:
        cn_norm = normalize_malayalam(cn.lower())
        if cn_norm in ASPECT_KEYWORDS:
            phrases.add(cn_norm)

    # Step 2: Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    model.config.output_hidden_states = True
    model.config.output_attentions = True

    outputs = model(inputs['input_ids'])
    mlm_output = outputs.logits  # or outputs[0] depending on model
    hidden_states = outputs.hidden_states
    attention_weights = outputs.attentions
    token_ids = inputs['input_ids'].squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    normalized_tokens = [normalize_malayalam(t) for t in tokens if t not in tokenizer.all_special_tokens]

    # Step 3: Top token indices based on embeddings
    top_token_indices = process_embeddings(hidden_states, inputs['input_ids'])

    # Step 4: Reconstruct full words from WordPiece tokens
    for idx in sorted(top_token_indices):
        token = tokens[idx]
        if token in SPECIAL_TOKENS or token in PUNCTUATIONS:
            continue
        clean_token = token.replace("▁", "").strip()
        if token.startswith("##"):
            current_word += clean_token
        else:
            if current_word:
                merged_tokens.append(current_word.strip())
            current_word = clean_token
    if current_word:
        merged_tokens.append(current_word.strip())

    # Step 5: Clean + Normalize + Lemmatize
    for token in merged_tokens:
        token = token.lower().strip(string.punctuation)

        if token in all_stopwords or token in SENTIMENT_WORDS:
            continue

        norm_token = normalize_malayalam(token)
        if norm_token in ASPECT_KEYWORDS:
            phrases.add(norm_token)
            continue

        # Lemmatize Malayalam/English
        if all('\u0D00' <= ch <= '\u0D7F' for ch in token):
            token = lemmatize_ml(token)
        else:
            token = lemmatize_en(token)

        norm_token = normalize_malayalam(token)

        for aspect in ASPECT_KEYWORDS:
            aspect_norm = normalize_malayalam(aspect)
            if norm_token == aspect_norm:
                phrases.add(aspect)
                break
        else:
            cleaned_tokens.append(token)

    cleaned_tokens = list(set(cleaned_tokens))

    # Step 6: Direct keyword match from original text
    for category, keywords in ASPECT_DICT.items():
        for keyword in keywords:
            norm_kw = normalize_malayalam(keyword)
            if re.search(r'\b' + re.escape(norm_kw) + r'\b', text):
                matched_aspects.add(keyword)

    phrases.update(matched_aspects)

    tokenlist=[]
    # Step 7: N-gram match (1 to 4-grams)
    max_n = min(4, len(cleaned_tokens))
    for n in range(max_n, 0, -1):
        for i in range(len(cleaned_tokens) - n + 1):
            ngram = " ".join(cleaned_tokens[i:i+n])
            norm_ngram = normalize_malayalam(ngram)
            tokenlist.append(norm_ngram)
            if norm_ngram in ASPECT_KEYWORDS:
                phrases.add(norm_ngram)
                continue

            for aspect in ASPECT_KEYWORDS:
                aspect_norm = normalize_malayalam(aspect)
                if norm_ngram == aspect_norm or norm_ngram.startswith(aspect_norm) or aspect_norm.startswith(norm_ngram):
                    phrases.add(aspect)
                    break

    # Step 8: Organize into aspect categories
    for phrase in phrases:
        phrase_norm = normalize_malayalam(phrase)

        for aspect, subterms in ASPECT_DICT.items():
            aspect_norm = normalize_malayalam(aspect)

            if phrase_norm == aspect_norm or phrase_norm.startswith(aspect_norm) or aspect_norm.startswith(phrase_norm):
                aspect_hits.setdefault(aspect, set())
                break

            for sub in subterms:
                sub_norm = normalize_malayalam(sub)
                if phrase_norm == sub_norm or phrase_norm.startswith(sub_norm) or sub_norm.startswith(phrase_norm):
                    aspect_hits.setdefault(aspect, set()).add(sub)
                    break

    # Step 9: Final output formatting
    final_output = {}
    for aspect in ASPECT_DICT:
        if aspect in aspect_hits:
            final_output[aspect] = sorted(aspect_hits[aspect]) if aspect_hits[aspect] else [aspect]

    return final_output, tokenlist

from transformers import BertForTokenClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-multilingual-cased"
model = BertForTokenClassification.from_pretrained(model_name).to(device)

sample_review = "ഞാൻ ആദ്യമായി ഒരു apple product വാങ്ങി. Apple productsil value for money എന്ന് തോന്നിയ ഒരേ ഒരു സാധനം..Ipad Air 5 with M1 processor. 50k..iOS nu ഒരുപാട് limitation ഉണ്ടെന്നു മനസിലായി."
tokens_review = word_tokenize(sample_review)
print(tokens_review)
inputs = tokenizer(sample_review, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

aspect_terms = extract_aspect_terms(sample_review)
print("Extracted Aspect Terms:", aspect_terms[0])

print(tokenizer(sample_review, return_tensors="pt"))


def extract_aspect_terms_iter(review_text):
    try:
        # Tokenize with truncation and max length handling
        inputs = tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Pass to the model or your extraction function
        aspect_terms = extract_aspect_terms(review_text)
        return aspect_terms
    except Exception as e:
        print(f"Error processing review: {review_text[:100]}... => {e}")
        return []

# Apply the extraction to each review in the DataFrame
df[['Extracted_Aspect_Terms', 'Word_Tokens']] = df['Review_Text'].apply(
    lambda x: pd.Series(extract_aspect_terms_iter(x))
)

# Optionally save to CSV
df.to_csv("aspect_terms_output.csv", index=False)

# Print sample output
print(df[["Review_Text", "Extracted_Aspect_Terms"]].head())

"""**checking validity of extracted aspect terms**"""

import pandas as pd
from collections import Counter

df_ATE = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/Conf ATE/ATE_updated.csv")
aspect_counter = Counter()
num_with_aspects = 0

for row in df_ATE['Updated_Aspect_Terms']:
    try:
        term_dict = eval(row)
        if term_dict:
            num_with_aspects += 1
        for terms in term_dict.values():
            aspect_counter.update(terms)
    except:
        continue

print(f"Coverage: {num_with_aspects}/{len(df_ATE)}")
print("Top extracted terms:", aspect_counter.most_common(10))

import ast
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
# Load your CSV
df_ATE = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/Conf ATE/ATE_updated.csv")


# Parse the dictionary column (if it's a string)
df_ATE['Updated_Aspect_Terms'] = df_ATE['Updated_Aspect_Terms'].fillna("{}").apply(ast.literal_eval)

# === 1. Coverage ===
df_ATE['Has_Aspect'] = df_ATE['Updated_Aspect_Terms'].apply(lambda x: len(x) > 0)
num_with_aspects = df_ATE['Has_Aspect'].sum()
total_reviews = len(df_ATE)

print(f"Total reviews: {total_reviews}")
print(f"Reviews with at least one extracted aspect: {num_with_aspects}")
print(f"Coverage: {100 * num_with_aspects / total_reviews:.2f}%")

# === 2. Frequency of Aspect Terms and Categories ===
aspect_counter = Counter()
category_counter = Counter()

for aspects in df_ATE['Updated_Aspect_Terms']:
    for category, terms in aspects.items():
        category_counter[category] += len(terms)
        aspect_counter.update(terms)

# Print top aspect categories
print("\nTop Aspect Categories:")
for cat, count in category_counter.most_common(10):
    print(f"{cat}: {count}")

# Print top aspect terms
print("\nTop Aspect Terms:")
for term, count in aspect_counter.most_common(10):
    print(f"{term}: {count}")

# === 3. Distribution / Skew Analysis ===
import seaborn as sns

# Convert counts to list for histogram
counts = list(category_counter.values())

plt.figure(figsize=(10, 5))
sns.histplot(counts, bins=15, kde=True)
plt.title("Distribution of Aspect Category Frequencies")
plt.xlabel("Frequency")
plt.ylabel("Number of Aspect Categories")
plt.show()

# Optionally, check underrepresented categories (e.g., < 3 mentions)
underrepresented = {cat: freq for cat, freq in category_counter.items() if freq < 3}
print(f"\nUnderrepresented Aspect Categories (<3 mentions): {len(underrepresented)}")
print(underrepresented)

import seaborn as sns
import pandas as pd

# Convert category_counter to DataFrame
cat_df = pd.DataFrame(category_counter.most_common(10), columns=["Category", "Frequency"])

plt.figure(figsize=(10, 6))
sns.barplot(data=cat_df, x="Frequency", y="Category", palette="Blues_d")
plt.title("Top 10 Aspect Categories")
plt.xlabel("Frequency")
plt.ylabel("Aspect Category")
plt.tight_layout()
plt.show()

aspect_freqs = aspect_counter  # or just use it directly
aspect_freqs_items = aspect_freqs.items()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

aspect_freqs = dict(sorted(aspect_freqs.items(), key=lambda x: x[1], reverse=True))  # Sort by frequency

# Convert to DataFrame for easier plotting
aspect_df = pd.DataFrame(list(aspect_freqs.items()), columns=['Aspect', 'Frequency'])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=aspect_df, x='Aspect', y='Frequency', palette='Blues_d')

# Annotate values
for index, row in aspect_df.iterrows():
    plt.text(index, row.Frequency + 1, str(row.Frequency), ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.title('Aspect Term Frequency Distribution')
plt.xlabel('Aspect Term / Category')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


!pip install --upgrade --force-reinstall numpy==1.24.4 scipy==1.9.3 gensim==4.3.1

import pandas as pd
import ast
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Load your CSV
df_ATE

# Parse aspect terms from Updated_Aspect_Terms
def extract_terms(row):
    try:
        term_dict = ast.literal_eval(row)
        terms = []
        for v in term_dict.values():
            terms.extend(v)
        return [t.lower() for t in terms]
    except:
        return []

# Extract all aspect terms as documents
docs = df_ATE['Updated_Aspect_Terms'].apply(extract_terms).tolist()

# Remove empty lists
docs = [doc for doc in docs if len(doc) > 0]

# Optionally remove stopwords (extend this list for Malayalam-English)
stop_words = set(load_stopwords('/content/merged_stopwords.txt'))
docs = [[word for word in doc if word not in stop_words] for doc in docs]

# Create Dictionary and Corpus
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Train LDA Model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10, random_state=42)

# Display topics
topics = lda_model.print_topics(num_words=5)
for idx, topic in topics:
    print(f"Topic {idx}: {topic}")


!pip install pyLDAvis

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis  # Use this for Gensim 4.x
import matplotlib.pyplot as plt

# Assuming your LDA model and corpus are already built
# lda_model -> your LdaModel
# corpus -> your document-term matrix
# dictionary -> your Gensim dictionary

# Prepare the visualization
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display in notebook
pyLDAvis.display(lda_vis)

# Optional: Save to HTML file
pyLDAvis.save_html(lda_vis, 'lda_visualization.html')

!apt-get -y install fonts-noto

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font family to a Malayalam-compatible font (if available)
rcParams['font.family'] = 'Noto Sans Malayalam'  # or another font you installed

# Use a transliteration function (simple)
def transliterate_malayalam(text):
    return text.encode('ascii', 'ignore').decode()

# Apply to words before plotting
words = [transliterate_malayalam(word) for word, _ in top_words]

# Assuming lda_model is already trained
# Set the number of top terms to display
topn = 5
num_topics = lda_model.num_topics  # Safe!
topn = 10
fig, axes = plt.subplots(1, num_topics, figsize=(5 * num_topics, 5), sharey=True)
axes = axes.flatten()

for idx, topic_id in enumerate(range(num_topics)):
    top_words = lda_model.show_topic(topic_id, topn=topn)
    words = [word for word, weight in top_words]
    weights = [weight for word, weight in top_words]

    axes[idx].barh(words[::-1], weights[::-1])
    axes[idx].set_title(f"Topic {topic_id}")
    axes[idx].tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.show()


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def aspect_coherence(aspect_dict, embed_fn):
    scores = {}
    for aspect, terms in aspect_dict.items():
        vectors = [embed_fn(term) for term in terms if term.strip()]
        if len(vectors) > 1:
            sims = cosine_similarity(vectors)
            mean_sim = np.mean(sims[np.triu_indices(len(sims), k=1)])
            scores[aspect] = mean_sim
    return scores

df_ATE = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/Conf ATE/updated_aspects.csv")


import ast
from collections import defaultdict

import ast
from collections import defaultdict

import ast
import re
from collections import defaultdict

import re
import difflib


def process_tokens(tokens):
    processed = []
    for tok in tokens:
        if not isinstance(tok, str):
            continue
        tok = tok.strip().lower()
        if not tok or tok in string.punctuation or tok.isdigit():
            continue

        # Lemmatize Malayalam/English
        if all('\u0D00' <= ch <= '\u0D7F' for ch in tok):
            tok = lemmatize_ml(tok)
        else:
            tok = lemmatize_en(tok)

        tok = normalize_malayalam(tok)


        if tok:  # skip empty results
            processed.append(tok)

    return processed

df_ATE["Lemmatized_Tokens"] = df["Word_Tokens"].apply(process_tokens)

df_ATE.to_csv("token_modi.csv", index=False)

def update_aspect_terms(row, aspect_dict):
    tokens = row['Word_Tokens']

    # Process tokens with normalization + lemmatization
    tokens = process_tokens(tokens)

    if isinstance(row['Extracted_Aspect_Terms'], str):
        current_aspects = ast.literal_eval(row['Extracted_Aspect_Terms'])
    else:
        current_aspects = row['Extracted_Aspect_Terms']

    updated_aspects = defaultdict(list, current_aspects)

    # Add missing matching keywords from tokens
    for category, keywords in aspect_dict.items():
        for kw in keywords:
            if kw in tokens and kw not in updated_aspects[category]:
                updated_aspects[category].append(kw)

    # Remove 'phone' from 'phone' category
    if "phone" in updated_aspects:
        updated_aspects["phone"] = [t for t in updated_aspects["phone"] if normalize(t) != "phone"]

    # Remove subword aspect terms within each category
    cleaned_aspects = {}
    for category, terms in updated_aspects.items():
        unique_terms = list(set(terms))
        filtered_terms = [t for t in unique_terms if not is_subword(t, unique_terms)]
        cleaned_aspects[category] = filtered_terms

    return dict(cleaned_aspects)

# Apply processing and save lemmatized tokens
df_ATE["Lemmatized_Tokens"] = df_ATE["Word_Tokens"].apply(process_tokens)
# Apply to the DataFrame
df_ATE['Updated_Aspect_Terms'] =df_ATE.apply(lambda row: update_aspect_terms(row, ASPECT_DICT), axis=1)

# Save updated version
df_ATE.to_csv("updated_aspects.csv", index=False)

df_aspect_terms_output = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/Conf ATE/aspect_terms_output.csv")

aspect_category_keywords = set()

for category, keywords in ASPECT_DICT.items():
    aspect_category_keywords.add(category)

print(aspect_category_keywords)

!python -m spacy download xx_ent_wiki_sm

import spacy
nlp = spacy.load("xx_ent_wiki_sm")

def get_ner_terms(text):
    doc = nlp(text)
    return [ent.text.lower().strip() for ent in doc.ents]

from difflib import SequenceMatcher

def is_fuzzy_match(a, b, threshold=0.8):
    """Returns True if similarity between a and b is >= threshold."""
    return SequenceMatcher(None, a, b).ratio() >= threshold

def map_ner_terms_to_aspects(ner_terms, existing_aspects, aspect_dict, use_fuzzy=False, fuzzy_threshold=0.8):
    updated_aspects = {k: list(v) for k, v in existing_aspects.items()}

    for term in ner_terms:
        term_lower = term.lower().strip()
        for aspect, keywords in aspect_dict.items():
            for kw in keywords:
                kw_lower = kw.lower().strip()
                if use_fuzzy:
                    # Use fuzzy matching
                    if is_fuzzy_match(term_lower, kw_lower, fuzzy_threshold):
                        if aspect not in updated_aspects:
                            updated_aspects[aspect] = []
                        if term not in updated_aspects[aspect]:
                            updated_aspects[aspect].append(term)
                        break
                else:
                    # Use exact match or substring matching
                    if term_lower == kw_lower or kw_lower in term_lower or term_lower in kw_lower:
                        if aspect not in updated_aspects:
                            updated_aspects[aspect] = []
                        if term not in updated_aspects[aspect]:
                            updated_aspects[aspect].append(term)
                        break
    return updated_aspects

"""**Graph-Based Refinement**"""

import pandas as pd
import networkx as nx
from collections import Counter
import community as community_louvain  # pip install python-louvain

# Load file 2 (flattened aspect terms per review)
df_terms = pd.read_csv('/content/drive/MyDrive/PHD_Corpus/Conf ATE/NERS.csv')  # columns: Review_Text, Aspect_Term, Aspect_Category

# Step 1: Group aspect terms by review
grouped = df_terms.groupby('Review_Text')['Aspect_Term'].apply(list)

# Build co-occurrence counts of terms within reviews
cooc_counts = Counter()
for term_list in grouped:
    unique_terms = set(term_list)
    for term1 in unique_terms:
        for term2 in unique_terms:
            if term1 != term2:
                # use sorted tuple as key to avoid duplicates (term1, term2) and (term2, term1)
                edge = tuple(sorted([term1, term2]))
                cooc_counts[edge] += 1

# Step 2: Build graph
G = nx.Graph()

# Add nodes
all_terms = set(df_terms['Aspect_Term'])
G.add_nodes_from(all_terms)

# Add edges with weights
for (term1, term2), weight in cooc_counts.items():
    G.add_edge(term1, term2, weight=weight)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Step 3: Community detection using Louvain method
partition = community_louvain.best_partition(G, weight='weight')

# Add community info as node attribute
nx.set_node_attributes(G, partition, 'community')

# Step 4: Compute centrality (e.g., PageRank)
pagerank = nx.pagerank(G, weight='weight')

# Attach pagerank scores to nodes
nx.set_node_attributes(G, pagerank, 'pagerank')

# Step 5: Rank terms by pagerank within each community
from collections import defaultdict

community_terms = defaultdict(list)
for node, comm in partition.items():
    community_terms[comm].append((node, pagerank[node]))

# Sort each community terms by pagerank descending
for comm in community_terms:
    community_terms[comm] = sorted(community_terms[comm], key=lambda x: x[1], reverse=True)

# Print top terms per community
for comm, terms in community_terms.items():
    print(f"Community {comm}:")
    for term, score in terms[:5]:  # top 5 terms
        print(f"  {term} (score: {score:.4f})")

import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
import community as community_louvain  # pip install python-louvain

# Load file 2 (flattened aspect terms per review)
df_terms = pd.read_csv('/content/drive/MyDrive/PHD_Corpus/Conf ATE/NERS.csv')  # columns: Review_Text, Aspect_Term, Aspect_Category

# Step 1: Group aspect terms by review
grouped = df_terms.groupby('Review_Text')['Aspect_Term'].apply(list)

# Build co-occurrence counts of terms within reviews
cooc_counts = Counter()
for term_list in grouped:
    unique_terms = set(term_list)
    for term1 in unique_terms:
        for term2 in unique_terms:
            if term1 != term2:
                edge = tuple(sorted([term1, term2]))
                cooc_counts[edge] += 1

# Step 2: Build graph
G = nx.Graph()
all_terms = set(df_terms['Aspect_Term'])
G.add_nodes_from(all_terms)
for (term1, term2), weight in cooc_counts.items():
    G.add_edge(term1, term2, weight=weight)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Step 3: Community detection
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'community')

# Step 4: Compute PageRank
pagerank = nx.pagerank(G, weight='weight')
nx.set_node_attributes(G, pagerank, 'pagerank')

# Step 5: Rank terms by PageRank within communities
community_terms = defaultdict(list)
for node, comm in partition.items():
    community_terms[comm].append((node, pagerank[node]))

# Sort and collect top terms per community
output_rows = []
for comm, terms in community_terms.items():
    sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
    for rank, (term, score) in enumerate(sorted_terms[:5], start=1):  # Top 5 per community
        output_rows.append({
            'Community': comm,
            'Rank': rank,
            'Aspect_Term': term,
            'PageRank_Score': round(score, 6)
        })

# Save to CSV
output_df = pd.DataFrame(output_rows)
output_df.to_csv('/content/drive/MyDrive/PHD_Corpus/Conf ATE/community_top_terms.csv', index=False)

print("Top community terms saved to 'community_top_terms.csv'")

import nltk

# Make sure NLTK tokenizer resources are downloaded:
nltk.download('punkt')

def create_bio_tags(review_text, candidate_aspects):
    review_text.apply(extract_aspect_terms_iter)
    tokens = nltk.word_tokenize(review_text)
    tags = ['O'] * len(tokens)
    lowered_tokens = [t.lower() for t in tokens]

    for aspect in candidate_aspects:
        aspect_tokens = nltk.word_tokenize(aspect.lower())
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if lowered_tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                tags[i] = 'B-ASP'
                for j in range(1, len(aspect_tokens)):
                    tags[i+j] = 'I-ASP'
    return tokens, tags


def rank_candidates_by_pagerank(graph, candidate_terms, top_n=10):
    pagerank = nx.get_node_attributes(graph, 'pagerank')
    filtered = {term: pagerank.get(term, 0) for term in candidate_terms if term in pagerank}
    ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return [term for term, score in ranked[:top_n]]

# Group aspect terms by review
grouped_aspects = df_terms.groupby('Review_Text')['Aspect_Term'].apply(list)

# Prepare lists for new columns
tokens_list = []
bio_tags_list = []

for review_text, aspects in grouped_aspects.items():
    ranked_terms = rank_candidates_by_pagerank(G, aspects, top_n=10)
    tokens, bio_tags = create_bio_tags(review_text, ranked_terms)

    tokens_list.append(tokens)
    bio_tags_list.append(bio_tags)

# Build new dataframe with Review_Text + Tokens + BIO_Tags
df_bio = pd.DataFrame({
    'Review_Text': grouped_aspects.index,
    'Tokens': tokens_list,
    'BIO_Tags': bio_tags_list
})

print(df_bio.head())



all_candidate_terms_per_review = []

for idx, row in df.iterrows():
    text = preprocess(row['Review_Text'])
    # NER extraction
    ner_terms = [ent[0] for ent in ner_extract(text)]
    # Token embeddings for semantic similarity and filtering could be added here
    # For demo: assume ner_terms as candidates
    candidates = semantic_filter(ner_terms)
    all_candidate_terms_per_review.append(candidates)

# Graph-based refinement on all reviews
refined_terms = refine_terms_with_graph(all_candidate_terms_per_review)

# Now generate BIO tags per review using refined terms
df['BIO_Tagged'] = df['Review_Text'].apply(lambda x: set_bio_tags(x, refined_terms))

# Print example
print(df[['Review_Text', 'BIO_Tagged']].head())

df.to_csv("BIO_Tag_v2.csv")

"""**Transliteration**"""

!pip install indic-transliteration

# import the module
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

text=df.Review_Text[5]
print(text,"\n")
# printing the transliterated text
transliterated=transliterate(text,sanscript.MALAYALAM,sanscript.ITRANS)
trans=str(transliterated.lower())
print(trans,"\n")

for i in range(28924,len(df)):
  text=df.Review_Text[i]
  print(text,"\n")
  # printing the transliterated text
  transliterated=transliterate(text,sanscript.MALAYALAM,sanscript.ITRANS)
  trans=str(transliterated.lower())
  print(trans,"\n")
  df.transliterated_reviews[i]=str(trans)

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, pipeline
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download('wordnet')

df = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/Balanced_CM.csv")  # Make sure this has a 'Review_Text' column

# Load spaCy model for POS tagging and dependency parsing
nlp = spacy.load('en_core_web_sm')  # English model for demonstration

# Load pretrained multilingual BERT and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")
model.eval()

# Load NER pipeline (can use multilingual or fine-tuned model if available)
ner_pipeline = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")

# Load your data
# df = pd.read_csv('code_mixed_reviews.csv')  # assume dataframe with 'Review_Text' column
texts = df['Review_Text'].tolist()

# Transliteration function Malayalam -> Latin
def transliterate_malayalam_to_latin(text):
    return transliterate(text, sanscript.MALAYALAM, sanscript.ITRANS)

# Step 1: Text Preprocessing & Normalization (including transliteration)
def preprocess_text(text):
    # Transliterate Malayalam script to Latin
    text = transliterate_malayalam_to_latin(text)
    # Additional normalization steps can be added here (lowercasing, removing URLs, special chars etc.)
    text = text.lower()
    return text

# Step 2: WordPiece Tokenization using BERT tokenizer
def tokenize_text(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_length - 2]
    input_ids = tokenizer.encode(tokens, return_tensors='pt', truncation=True, max_length=max_length, add_special_tokens=True)
    return tokens, input_ids

# Step 3: Named Entity Recognition (NER)
def perform_ner(text):
    ner_results = ner_pipeline(text)
    entities = [(ent['word'], ent['entity_group']) for ent in ner_results]
    return entities

# Step 4: Dependency Parsing and POS tagging using spaCy
def dependency_pos_parse(text):
    doc = nlp(text)
    return [(token.text, token.pos_, token.dep_, token.head.text) for token in doc]

# Step 5: Lexical-Semantic Resource Lookup using WordNet (example)
def lexical_semantic_lookup(token):
    synsets = wn.synsets(token)
    lemmas = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            lemmas.add(lemma.name())
    return lemmas

# Step 6: Extract embeddings from pre-trained multilingual transformer
def get_bert_embeddings(text, max_length=512):
    tokens, input_ids = tokenize_text(text, max_length)
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
    return tokens, embeddings

# Step 7: Unsupervised Aspect Term Extraction
def extract_aspect_terms(texts, n_clusters=50):
    all_tokens = []
    all_embeddings = []

    for review in texts:
        preprocessed_text = preprocess_text(review)
        tokens, embeddings = get_bert_embeddings(preprocessed_text)
        for tok, emb in zip(tokens, embeddings):
            if tok.startswith('##'):  # skip subwords for clustering
                continue
            all_tokens.append(tok)
            all_embeddings.append(emb.numpy())

    # Cluster tokens by embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(all_embeddings)
    token_cluster_map = {token: cluster for token, cluster in zip(all_tokens, kmeans.labels_)}

    # Select frequent tokens per cluster as aspect candidates
    cluster_tokens = defaultdict(list)
    for token, cluster in token_cluster_map.items():
        cluster_tokens[cluster].append(token)

    aspect_term_candidates = set()
    for cluster, tokens in cluster_tokens.items():
        common = Counter(tokens).most_common(5)
        for token, _ in common:
            aspect_term_candidates.add(token.lower())

    return aspect_term_candidates

# Step 8: BIO Tagging of aspect terms in text
def bio_tagging(text, aspect_terms):
    preprocessed_text = preprocess_text(text)
    tokens = word_tokenize(preprocessed_text)
    tags = ['O'] * len(tokens)

    i = 0
    while i < len(tokens):
        if tokens[i].lower() in aspect_terms:
            tags[i] = 'B-ASP'
            j = i + 1
            while j < len(tokens) and tokens[j].lower() in aspect_terms:
                tags[j] = 'I-ASP'
                j += 1
            i = j
        else:
            i += 1
    return list(zip(tokens, tags))

# Run the full workflow on the dataset
print("Starting unsupervised aspect term extraction workflow...")

# Extract aspect term candidates from entire corpus
aspect_candidates = extract_aspect_terms(texts)

# Annotate each review with BIO tagging
df['BIO_Annotated'] = df['Review_Text'].apply(lambda x: bio_tagging(x, aspect_candidates))

# Example: perform NER, POS, dependency and lexical lookup on first review
sample_text = texts[0]
print("\nSample review:", sample_text)

print("\nNER results:")
print(perform_ner(preprocess_text(sample_text)))

print("\nDependency and POS parse:")
print(dependency_pos_parse(preprocess_text(sample_text)))

print("\nLexical-semantic lookup for token 'phone':")
print(lexical_semantic_lookup('phone'))

# Save annotated results
df.to_csv("bio_annotated_aspect_terms.csv", index=False)
print("\nBIO annotated dataset saved.")

# Step 9: Evaluation & Comparison of Models (to be implemented)
# Placeholder: here you would run different ATE models, compute precision, recall, F1 on a small manually annotated subset or by indirect metrics.
print("\nEvaluation & comparison step: Implement model evaluations as per your annotated validation data.")

# Step 5: Cluster Tokens into Aspect Term Candidates
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(all_embeddings)

token_cluster_map = {token: cluster for token, cluster in zip(all_tokens, kmeans.labels_)}

# Step 6: Identify Frequent Tokens per Cluster
from collections import Counter, defaultdict

cluster_tokens = defaultdict(list)
for token, cluster in token_cluster_map.items():
    cluster_tokens[cluster].append(token)

# Keep top-n tokens per cluster as aspect term candidates
aspect_term_candidates = set()
for cluster, tokens in cluster_tokens.items():
    common = Counter(tokens).most_common(5)
    for token, _ in common:
        aspect_term_candidates.add(token.lower())

print("Sample aspect term candidates:", list(aspect_term_candidates)[:10])

# Step 7: BIO Tagging Heuristics
def bio_tagging(text, aspect_terms):
    tokens = word_tokenize(text)
    tags = ['O'] * len(tokens)
    for i, token in enumerate(tokens):
        if token.lower() in aspect_terms:
            tags[i] = 'B-ASP'
            if i + 1 < len(tokens) and tokens[i+1].lower() in aspect_terms:
                tags[i+1] = 'I-ASP'
    return list(zip(tokens, tags))

# Step 8: Apply BIO Tagging
df['BIO_Annotated'] = df['Review_Text'].apply(lambda x: bio_tagging(x, aspect_term_candidates))

# Save Output
df.to_csv("bio_annotated_aspect_terms.csv", index=False)
print("BIO-annotated dataset saved.")
