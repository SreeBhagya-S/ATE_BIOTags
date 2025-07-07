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

class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()


    def forward(self, src, attention_mask=None):
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # mask where attention_mask is 0 (padding)

        attn_output, attn_weights = self.self_attn(
            src, src, src, key_padding_mask=key_padding_mask, need_weights=True
        )
        src2 = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.norm2(src2 + self.dropout2(ff_output))
        return src2, attn_weights



class TransformerEncoderModel(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderModel, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.transformer_layers = nn.ModuleList([
        TransformerEncoderLayerWithAttn(config.d_model, config.nhead, config.dim_feedforward, dropout=0.2)
        for _ in range(config.num_layers)
        ])
        self.fc = nn.Linear(config.d_model, config.vocab_size)  # MLM Head
        self.nsp_fc = nn.Linear(config.d_model, 2)  # NSP Head
        self.dropout = nn.Dropout(0.1)
        self.max_seq_length = config.max_seq_length
        self.output_hidden_states = config.output_hidden_states  
        self.absa_fc = nn.Linear(config.d_model, config.num_absa_classes)



    def forward(self, src, attention_mask=None, return_hidden_states=None, return_attentions=False):
        if return_hidden_states is None:
            return_hidden_states = self.output_hidden_states

        src = self.embedding(src) * (self.max_seq_length ** 0.5)
        src = self.dropout(src)

        attention_weights = []
        hidden_states = []

        x = src
        for layer in self.transformer_layers:
            x, attn = layer(x, attention_mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
            if return_attentions:
                attention_weights.append(attn)

        output = x  # Final output from last layer
        mlm_output = self.fc(output)
        cls_token = output[:, 0, :]  # CLS for NSP
        nsp_output = self.nsp_fc(cls_token)

        results = [mlm_output, nsp_output]
        if return_hidden_states:
            results.append(output)
        if return_attentions:
            results.append(attention_weights)

        absa_logits = self.absa_fc(output[:, 0, :])  # [CLS] token
        results.append(absa_logits)

        return tuple(results)


    def extract_embeddings(self, src, output_hidden_states=False):
        """Returns word embeddings, contextualized embeddings, and sentence embeddings."""
        with torch.no_grad():
            word_embeddings = self.embedding(src)
            word_embeddings = self.dropout(word_embeddings)

            transformer_input = word_embeddings.permute(1, 0, 2)
            hidden_states = []
            x = transformer_input
            for layer in self.transformer_encoder.layers:
                x = layer(x)
                if output_hidden_states:
                    hidden_states.append(x.permute(1, 0, 2))
            x = x.permute(1, 0, 2)
            sentence_embeddings = x[:, 0, :]

        if output_hidden_states:
            return word_embeddings, x, sentence_embeddings, hidden_states
        return word_embeddings, x, sentence_embeddings

class TransformerConfig:
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, output_hidden_states=True, num_absa_classes=3):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        self.output_hidden_states = output_hidden_states
        self.num_absa_classes = num_absa_classes 

    @classmethod
    def from_pretrained(cls, config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

config_path = "/content/drive/MyDrive/PHD_Corpus/7Jan_model_V2_config_14l_8BS_512SQ_100E_LR1e-4_TS25_RS5/7Jan_config_V2_14l_8BS_512SQ_100E_LR1e-4_TS25_RS5_12AH.json"
config = TransformerConfig.from_pretrained(config_path)
config.num_absa_classes = 5  # or 5, based on your task

model = TransformerEncoderModel(config)


print(model)


from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/content/drive/MyDrive/PHD_Corpus/tokenizer.json")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load config and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device,"alloted..............................................")
model = TransformerEncoderModel(config).to(device)

# Load MLM checkpoint (trained on MLM+NSP)
checkpoint_path = "/content/drive/MyDrive/PHD_Corpus/7Jan_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

from torch.utils.data import Dataset

class AspectSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=128):
        self.tokenizer = tokenizer
        self.texts = df['Review_Text']
        self.aspects = df['Aspect_Term']
        self.categories = df['Aspect_Category']
        self.labels = df['Sentiment_Class'].map(label2id)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"{self.texts.iloc[idx]} [ASP] {self.aspects.iloc[idx]} [CAT] {self.categories.iloc[idx]}"
        encoded = self.tokenizer(
            text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        return (
            encoded['input_ids'].squeeze(),
            encoded['attention_mask'].squeeze(),
            torch.tensor(self.labels.iloc[idx])
        )

from torch.utils.data import DataLoader, random_split

dataset = AspectSentimentDataset(df, tokenizer, label2id, max_len=128)

# Split into train and val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

num_epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        print(f"Batch: {batch}")
        print(f"Input IDs shape: {batch[0].shape}")
        print(f"Attention mask shape: {batch[1].shape}")
        print(f"Labels shape: {batch[2].shape}")

        # Filter out rows with NaN labels
        valid_mask = torch.isnan(batch[2]) == False
        input_ids = batch[0][valid_mask]
        attention_mask = batch[1][valid_mask]
        labels = batch[2][valid_mask]

        # Convert labels to the correct type
        labels = labels.to(torch.long)

        # Move to device
        input_ids, attention_mask, labels = [b.to(device) for b in [input_ids, attention_mask, labels]]


        # input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        print(f"Model output shape: {outputs[0].shape}")

        absa_logits = outputs[-1]  # ABSA logits
        print(absa_logits.dtype)
        # Ensure absa_logits is of type torch.float32
        absa_logits = absa_logits.float()
        print(absa_logits.shape)
        print(labels.min(), labels.max())

        # Ensure labels are of type torch.long (for classification)
        labels = labels.long()

        # Compute the loss
        loss = loss_fn(absa_logits, labels)


        # loss = loss_fn(absa_logits, labels)  # labels: shape [batch_size]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def preprocess_absa_data(df, tokenizer, label2id, max_len=128):
    input_ids, attention_masks, labels = [], [], []

    for _, row in df.iterrows():
        text = f"{row['Review_Text']} [ASP] {row['Aspect_Term']} [CAT] {row['Aspect_Category']}"
        encoded = tokenizer.encode_plus(
            text, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        labels.append(label2id[row['Sentiment_Class']])

    return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[-1]  # aspect_sentiment_output

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



def tokenize_review_aspect(review_text, aspect_term, aspect_category, tokenizer, max_length=512):
    # Combine review text, aspect term, and aspect category
    input_text = f"Review: {review_text} Aspect: {aspect_term} Category: {aspect_category}"

    # Tokenize the combined input text
    encoding = tokenizer(input_text, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    return encoding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dictionary to map label indices to string (edit based on your label mapping)
id2label = {
    0: "Positive",
    1: "Negative",
    2: "Neutral",
    3: "Mixed Feelings",
    4: "Not relevant"
}

# Store output rows here
output_rows = []

# Process each row
for index, row in df.iterrows():
    review_text = row['Review_Text']
    aspect_term = row['Aspect_Term']
    aspect_category = row['Aspect_Category']

    encoding = tokenize_review_aspect(review_text, aspect_term, aspect_category, tokenizer)
    sentence_embedding, attention_weights = get_embeddings_and_attention(model, encoding, device=device)
    pred_sent_id = classify_aspect_sentiment(model, sentence_embedding)
    pred_sent_label = id2label[pred_sent_id]

    output_rows.append({
        "Review_Text": review_text,
        "Aspect_Term": aspect_term,
        "Aspect_Category": aspect_category,
        "Predicted_Aspect_Sentiment": pred_sent_label
    })

# Save the results to a new CSV
output_df = pd.DataFrame(output_rows)
output_df.to_csv("predicted_absa_results.csv", index=False)
print("Saved predicted results to predicted_absa_results.csv")

aspect_outputs = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    review = row["Review_Text"]
    aspect = row["Aspect_Term"]

    # Replace exact match (case-insensitive) of aspect term with special tokens
    def mark_aspect_in_text(text, aspect):
        return text.replace(aspect, f"[ASPECT] {aspect} [/ASPECT]")

    marked_text = mark_aspect_in_text(review, aspect)

    # Tokenize
    encoded = tokenizer(
        marked_text,
        return_tensors='pt',
        padding='max_length',
        max_length=512,
        truncation=True,
        add_special_tokens=True
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        _, _, hidden_states = model(input_ids, attention_mask=attention_mask, return_hidden_states=True)

    # Decode to find exact token indices of the aspect term (between [ASPECT] and [/ASPECT])
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    try:
        start_idx = tokens.index("[ASPECT]") + 1
        end_idx = tokens.index("[/ASPECT]")
        aspect_indices = list(range(start_idx, end_idx))
    except ValueError:
        # [ASPECT] or [/ASPECT] not found, skip
        continue

    # Extract context-aware embedding for the aspect term
    aspect_embedding = hidden_states[0, aspect_indices, :].mean(dim=0).cpu().numpy()

    aspect_outputs.append({
        "Review_Text": review,
        "Aspect_Term": aspect,
        "Aspect_Category": row["Aspect_Category"],
        "Aspect_Embedding": aspect_embedding
    })

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Stack all the aspect embeddings into a 2D numpy array (samples x features)
aspect_embeddings = np.array([output['Aspect_Embedding'] for output in aspect_outputs])

# KMeans clustering (choose number of clusters as needed)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(aspect_embeddings)

# Add the cluster labels to the outputs for later analysis
for idx, output in enumerate(aspect_outputs):
    output["Cluster_Label"] = clusters[idx]

# Optional: Visualize with PCA (reduce dimensions for visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(aspect_embeddings)

# Plot the results (if you want to visualize them)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis', marker='o')
plt.colorbar()
plt.title("PCA of Aspect Term Embeddings with Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Optional: Export results to CSV
import pandas as pd

df_embeddings = pd.DataFrame(aspect_outputs)
df_embeddings.to_csv('aspect_term_embeddings_with_clusters.csv', index=False)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Convert list of embeddings to matrix
X = np.stack(aspect_embedding)

# Cluster with KMeans (try 3 for positive/negative/neutral-like groupings)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualization
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='coolwarm', s=30)
plt.title("Aspect Term Embeddings Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.show()

# Optional: See results with terms
df_clustered = pd.DataFrame({
    'Aspect_Term': aspect,
    'Cluster_ID': clusters
})
print(df_clustered.head(20))

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Scale the features
X_scaled = StandardScaler().fit_transform(aspect_embedding)

# Run DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='tab10', s=30)
plt.title("DBSCAN Clustering of Aspect Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.colorbar(label="Cluster ID")
plt.show()

# Save to DataFrame
df_dbscan = pd.DataFrame({
    'Aspect_Term': aspect,
    'Cluster_ID': dbscan_labels
})

from sklearn.metrics import silhouette_score

silhouette_scores = []
range_n_clusters = list(range(2, 10))

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    score = silhouette_score(X, cluster_labels)
    silhouette_scores.append(score)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different K (KMeans)")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# If you want to export the DBSCAN results
df_dbscan.to_csv("aspect_clusters_dbscan.csv", index=False)

# If you used KMeans instead:
df_kmeans = pd.DataFrame({
    'Aspect_Term': aspect,
    'Cluster_ID': clusters
})
df_kmeans.to_csv("aspect_clusters_kmeans.csv", index=False)




# Define the model architecture (correcting the forward pass handling)
class TransformerForAspectSentimentClassification(nn.Module):
    def __init__(self, base_model, num_classes):
        super(TransformerForAspectSentimentClassification, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Access the embedding layer from the base model
        embedding_layer = base_model.embedding
        vocab_size = embedding_layer.num_embeddings
        embedding_dim = embedding_layer.embedding_dim

        # Initialize classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)  # Use the embedding dimension as input size

    def forward(self, input_ids, attention_mask):
        # Forward pass through the base model
        _, _, hidden_states, attentions = self.base_model(
            input_ids, attention_mask,
            return_hidden_states=True,
            return_attentions=True
        )

        cls_hidden_state = hidden_states[:, 0, :]  # [CLS] token

        logits = self.classifier(cls_hidden_state)     # [batch_size, num_classes]


        return logits, cls_hidden_state, hidden_states, attentions

# Dataset class to handle inputs correctly
class AspectSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, label2id):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

        from tokenizers.processors import TemplateProcessing
        # Set up post-processor using your custom tokens
        sep_token = "<SEP>"
        sep_token_id = tokenizer.token_to_id(sep_token)

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {sep_token}",
            pair=f"$A {sep_token} $B:1",
            special_tokens=[
                (sep_token, sep_token_id),
            ]
        )
    def __getitem__(self, idx):
        review = self.df.iloc[idx]['Review_Text']
        aspect = self.df.iloc[idx]['Aspect_Term']
        sentiment = self.df.iloc[idx]['Sentiment_Class']

        input_text = f"{review} [SEP] {aspect}"

        try:
            encoding = self.tokenizer.encode(input_text)
            input_ids = encoding.ids[:self.max_length]
            attention_mask = encoding.attention_mask[:self.max_length]

            padding_len = self.max_length - len(input_ids)
            if padding_len > 0:
                input_ids += [0] * padding_len
                attention_mask += [0] * padding_len

        except Exception as e:
            print(f"Encoding error at idx {idx}: {e}")
            input_ids = [0] * self.max_length
            attention_mask = [0] * self.max_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'sentiment_class': torch.tensor(self.label2id.get(sentiment, -1), dtype=torch.long),
            'review_text': review,
            'aspect_term': aspect
        }




    def __len__(self):
        return len(self.df)

# Hyperparameters and config
max_length = 128  # Adjust as needed
label2id = {
    'Positive': 0,
    'Negative': 1,
    'Neutral': 2,
    'Mixed Feelings': 3,
    'Not_relevant': 4
}

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/absa_transformed.csv")

# Mapping Mixed Feelings and Not Relevant to Neutral
sentiment_map = {
    'Mixed Feelings': 'Neutral',
    'Not Relevant': 'Neutral'
}

from sklearn.model_selection import train_test_split


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Sentiment_Class'])

train_df['Sentiment_Class'] = train_df['Sentiment_Class'].map(sentiment_map).fillna(train_df['Sentiment_Class'])

train_dataset = AspectSentimentDataset(train_df, tokenizer, max_length, label2id)
val_dataset = AspectSentimentDataset(val_df, tokenizer, max_length, label2id)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model initialization
classification_model = TransformerForAspectSentimentClassification(base_model=model, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classification_model.parameters(), lr=1e-5)

from tqdm import tqdm

best_val_loss = float('inf')
patience = 5
patience_counter = 0
num_epochs = 20

predicted_aspects = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    classification_model.train()
    total_train_loss = 0

    train_loop = tqdm(train_loader, desc="Training", leave=False)
    for batch in train_loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['sentiment_class'].to(device)

        optimizer.zero_grad()
        logits, _, _, _ = classification_model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_loop.set_postfix(train_loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    classification_model.eval()
    total_val_loss = 0
    val_loop = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in val_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['sentiment_class'].to(device)

            logits, _, _, _ = classification_model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            val_loop.set_postfix(val_loss=loss.item())

            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            for i in range(len(predictions)):
                predicted_aspects.append({
                    'Review_Text': batch['review_text'][i],
                    'Aspect_Term': batch['aspect_term'][i],
                    'Predicted_Label': predictions[i]
                })

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"\n📊 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(classification_model.state_dict(), "best_model.pt")
        print("✅ New best model saved.")
    else:
        patience_counter += 1
        print(f"⚠️ No improvement. Patience used: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

# Save predictions
predicted_df = pd.DataFrame(predicted_aspects)
predicted_df['Predicted_Label'] = predicted_df['Predicted_Label'].map({v: k for k, v in label2id.items()})
predicted_df.to_csv("aspect_sentiment_predictions.csv", index=False)
print("\n✅ Predictions saved to 'aspect_sentiment_predictions.csv'")



label2id = {
    'Positive': 0,
    'Negative': 1,
    'Neutral': 2,
    'Mixed Feelings': 3,
    'Not_relevant': 4
}

# Fix for KeyError
def get_sentiment_class(sentiment_class):
    sentiment_class = sentiment_class.strip()  # Ensure no leading/trailing whitespaces
    if sentiment_class in label2id:
        return label2id[sentiment_class]
    else:
        print(f"Unknown sentiment class: {sentiment_class}")
        return -1  # Return a default value (or handle accordingly)


df = pd.read_csv("/content/drive/MyDrive/PHD_Corpus/absa_transformed.csv")
print(df.columns.tolist())

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.2,            # 80/20 split
    random_state=42,
    stratify=df['Sentiment_Class']  # ensure class distribution remains similar
)



from torch.utils.data import DataLoader

# Initialize tokenizers and max length
max_length = 128  # Adjust as per your needs


# Train and test datasets
train_dataset = AspectSentimentDataset(train_df, tokenizer, max_length,label2id)
test_dataset = AspectSentimentDataset(val_df, tokenizer, max_length,label2id)

# Dataloaders for batching during training and testing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


import pandas as pd
from torch import optim
import torch.nn as nn
import torch
import numpy as np

# --- Prediction Mapping Function ---
def map_predictions_to_aspects(logits, aspect_terms):
    predictions = torch.argmax(logits, dim=-1)
    aspect_sentiments = [pred.item() for pred in predictions]
    return aspect_sentiments

# --- Model Setup ---
num_classes = 3  # Positive, Negative, Neutral
classification_model = TransformerForAspectSentimentClassification(base_model=model, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classification_model.parameters(), lr=1e-5)

best_val_loss = float('inf')
patience = 5
patience_counter = 0
num_epochs = 20

# To store predictions for final CSV
predicted_aspects = []

# --- Training Loop ---
for epoch in range(num_epochs):
    classification_model.train()
    total_train_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits, _, _, _ = classification_model(input_ids, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation Phase ---
    classification_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, _, _, _ = classification_model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(predictions)):
                predicted_aspects.append({
                    'Review_Text': batch['review_text'][i],
                    'Aspect_Term': batch['aspect_term'][i],
                    'Predicted_Label': predictions[i]
                })

    avg_val_loss = total_val_loss / len(test_loader)
    print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(classification_model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        print(f'Validation loss did not improve. Patience: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# --- Save Predictions to CSV ---
predicted_df = pd.DataFrame(predicted_aspects)

# If you want human-readable sentiment labels (reverse map)
# Example: id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
predicted_df['Predicted_Label'] = predicted_df['Predicted_Label'].map(label2id)

predicted_df.to_csv("aspect_sentiment_predictions.csv", index=False)

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

"""**code for Coverage and Frequency Statistics**"""

# # Parse the dictionary column (if it's a string)
# def safe_parse(val):
#     if isinstance(val, dict):
#         return val
#     try:
#         return ast.literal_eval(val)
#     except:
#         return {}

# df_ATE['Updated_Aspect_Terms'] = df_ATE['Updated_Aspect_Terms'].apply(safe_parse)


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

df_bio.to_csv("BIO_Tag.csv")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter

# Load multilingual BERT and spaCy model (English + custom Malayalam POS, dependency if available)
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

# Load spaCy model for POS and dependency parsing (use 'en_core_web_sm' for demo)
nlp = spacy.load('en_core_web_sm')

# Sample bilingual stopwords set (add your lists)
stopwords = STOPWORDS

# Aspect seed lexicon example (expand this)
ASPECT_SEEDS = ASPECT_DICT

def preprocess(text):
    # Add your script detection and Unicode normalization here
    return text.lower()

def tokenize(text):
    return tokenizer.tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stopwords]

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    return tokens, embeddings, inputs["offset_mapping"].squeeze(0)

def ner_extract(text):
    # Use your NER model here. Demo: extract named entities via spaCy
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def semantic_filter(candidates):
    # Filter by POS, lexicon, etc.
    filtered = []
    for term in candidates:
        doc = nlp(term)
        # Only keep if head POS is noun or proper noun
        if any(tok.pos_ in ['NOUN', 'PROPN'] for tok in doc):
            filtered.append(term)
    return filtered

def build_cooccurrence_graph(grouped_terms):
    cooc_counts = Counter()
    for term_list in grouped_terms:
        unique_terms = set(term_list)
        for term1 in unique_terms:
            for term2 in unique_terms:
                if term1 != term2:
                    edge = tuple(sorted([term1, term2]))
                    cooc_counts[edge] += 1
    G = nx.Graph()
    all_terms = set(term for sublist in grouped_terms for term in sublist)
    G.add_nodes_from(all_terms)
    for (term1, term2), weight in cooc_counts.items():
        G.add_edge(term1, term2, weight=weight)
    return G

def refine_terms_with_graph(terms_per_review):
    G = build_cooccurrence_graph(terms_per_review)
    partition = community_louvain.best_partition(G, weight='weight')
    pagerank = nx.pagerank(G, weight='weight')
    nx.set_node_attributes(G, partition, 'community')
    nx.set_node_attributes(G, pagerank, 'pagerank')

    community_terms = defaultdict(list)
    for node, comm in partition.items():
        community_terms[comm].append((node, pagerank[node]))
    for comm in community_terms:
        community_terms[comm] = sorted(community_terms[comm], key=lambda x: x[1], reverse=True)
    # Return top N terms per community
    refined_terms = set()
    for comm in community_terms:
        top_terms = [term for term, _ in community_terms[comm][:5]]
        refined_terms.update(top_terms)
    return refined_terms

def set_bio_tags(text, aspect_terms):
    tokens = tokenize(text)
    bio_tags = ['O'] * len(tokens)
    text_lower = text.lower()
    # map tokens back to character spans for matching aspect terms (approximate)
    for aspect in aspect_terms:
        aspect_tokens = tokenize(aspect)
        # naive substring matching in token list
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                bio_tags[i] = 'B-ASP'
                for j in range(i+1, i+len(aspect_tokens)):
                    bio_tags[j] = 'I-ASP'
    return list(zip(tokens, bio_tags))

# ----------------------------
# Main pipeline on your DataFrame
# ----------------------------

# Assume df with columns: 'Review_Text'
df

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
