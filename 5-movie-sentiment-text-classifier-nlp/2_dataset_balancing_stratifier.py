"""
Dataset Aggregation & Stratification Pipeline
Author: Meet Ahalpara

This module serves as the critical data balancing and stratification layer for the 
4-class sentiment classifier. It merges authentic long-form IMDB reviews with 
generated synthetic data (short, mixed, and neutral) to construct a mathematically 
balanced, robust dataset. Finally, it executes a strict stratified split to ensure 
uniform class distributions across training and evaluation boundaries.
"""
import os
import random
import re
import pandas as pd

random.seed(42)

print("\n" + "="*65)
print(" NLP DATASET AGGREGATION & STRATIFICATION PIPELINE INITIATED")
print("="*65)

# ==========================================
# PHASE 1: ENVIRONMENT & PATH RESOLUTION
# ==========================================
print("\n[INFO] Resolving directory paths and validating foundational assets...")
# The absolute directory paths are dynamically resolved to ensure the pipeline remains portable.
# The system strictly validates the presence of required intermediate files before proceeding.
project_folder = os.path.dirname(os.path.abspath(__file__))

dataset_folder = os.path.join(project_folder, "Dataset")
review_dataset_folder = os.path.join(dataset_folder, "ReviewDatasets")
final_dataset_folder = os.path.join(dataset_folder, "FinalDatasets")

csv_dataset_path = os.path.join(dataset_folder, "IMDB Dataset.csv")

os.makedirs(final_dataset_folder, exist_ok=True)

positive_short_path = os.path.join(review_dataset_folder, "positive_short_reviews.csv")
negative_short_path = os.path.join(review_dataset_folder, "negative_short_reviews.csv")
mixed_path = os.path.join(review_dataset_folder, "mixed_reviews.csv")
neutral_path = os.path.join(review_dataset_folder, "neutral_reviews.csv")

final_full_path = os.path.join(final_dataset_folder, "final_dataset.csv")
final_train_path = os.path.join(final_dataset_folder, "final_train.csv")
final_test_path = os.path.join(final_dataset_folder, "final_test.csv")

required_files = [
    csv_dataset_path,
    positive_short_path,
    negative_short_path,
    mixed_path,
    neutral_path
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

print(f"[SUCCESS] All {len(required_files)} requisite dataset partitions verified.")

# ==========================================
# PHASE 2: AUTHENTIC DATA INGESTION
# ==========================================
print("\n[INFO] Loading foundational IMDB dataset for long-form extraction...")
# The foundational authentic dataset is loaded to extract complex, long-form linguistic patterns.
df = pd.read_csv(csv_dataset_path)

print(f"[SUCCESS] Authentic dataset mapped. Detected Columns: {df.columns.tolist()}")

text_col = "review"
label_col = "sentiment"

# ==========================================
# PHASE 3: TEXT SANITIZATION PROTOCOLS
# ==========================================
print("\n[INFO] Applying rigorous text sanitization to the authentic corpus...")

def clean_text(text):
    """
    Standardizes raw input text to guarantee mathematical uniformity during subsequent tokenization.
    
    Args:
        text (str): Raw review text.
        
    Returns:
        str: A lowercased string rigorously stripped of HTML markup, non-alphanumeric 
             characters, and irregular whitespace distributions.
    """
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)  # Replace HTML line breaks with space 
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text) # Remove special characters, keep only letters, numbers, and spaces
    text = re.sub(r"\s+", " ", text).strip() # Replace multiple spaces with a single space and trim leading/trailing spaces
    return text

df[text_col] = df[text_col].apply(clean_text)

print("[SUCCESS] Authentic corpus successfully sanitized.")

# ==========================================
# PHASE 4: DATASET MERGING & CLASS BALANCING
# ==========================================
print("\n[INFO] Extracting and merging authentic and synthetic class matrices...")

# Anchor Subset Extraction: A perfectly balanced subset of 1000 authentic reviews per class 
# is extracted to anchor the model's understanding of complex, long-form natural language.
positive_real_df = df[df[label_col].str.lower() == "positive"][[text_col]].copy()
negative_real_df = df[df[label_col].str.lower() == "negative"][[text_col]].copy()

positive_real_df = positive_real_df.sample(n=1000, random_state=42).copy()
negative_real_df = negative_real_df.sample(n=1000, random_state=42).copy()

positive_real_df = positive_real_df.rename(columns={text_col: "text"})
negative_real_df = negative_real_df.rename(columns={text_col: "text"})

positive_real_df["sentiment"] = "positive"
negative_real_df["sentiment"] = "negative"

# Synthetic Injection: 1000 synthetic short-form reviews are injected per class. 
# This hybrid structural approach guarantees the model generalizes across varying 
# sentence lengths and polarity densities.
positive_short_df = pd.read_csv(positive_short_path)
negative_short_df = pd.read_csv(negative_short_path)

positive_short_df = positive_short_df[["text", "sentiment"]].sample(n=1000, random_state=42).copy()
negative_short_df = negative_short_df[["text", "sentiment"]].sample(n=1000, random_state=42).copy()

positive_short_df["text"] = positive_short_df["text"].apply(clean_text)
negative_short_df["text"] = negative_short_df["text"].apply(clean_text)

# Aggregation: Authentic and synthetic subsets are concatenated and shuffled to 
# finalize exactly 2000 positive and 2000 negative structural samples.
positive_df = pd.concat([positive_real_df, positive_short_df], ignore_index=True)
negative_df = pd.concat([negative_real_df, negative_short_df], ignore_index=True)

positive_df = positive_df.sample(frac=1, random_state=42).reset_index(drop=True)
negative_df = negative_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Expansion: 2000 synthetic reviews are sampled for both the 'mixed' and 'neutral' classes
# to ensure a perfectly uniform 4-way class distribution across 8000 total training boundaries.
mixed_df = pd.read_csv(mixed_path)
neutral_df = pd.read_csv(neutral_path)

mixed_df = mixed_df[["text", "sentiment"]].sample(n=2000, random_state=42).copy()
neutral_df = neutral_df[["text", "sentiment"]].sample(n=2000, random_state=42).copy()

mixed_df["text"] = mixed_df["text"].apply(clean_text)
neutral_df["text"] = neutral_df["text"].apply(clean_text)

print(f"[SUCCESS] Class equilibrium strictly achieved.")
print(f"|-- Aggregated Positive Vector : {len(positive_df)}")
print(f"|-- Aggregated Negative Vector : {len(negative_df)}")
print(f"|-- Aggregated Mixed Vector    : {len(mixed_df)}")
print(f"|-- Aggregated Neutral Vector  : {len(neutral_df)}")

# Master Matrix Formation: The master DataFrame is established and rigorously shuffled 
# to prevent cyclical backpropagation biases during downstream neural network training.
final_df = pd.concat(
    [positive_df, negative_df, mixed_df, neutral_df],
    ignore_index=True
)

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n[PREVIEW] Consolidated Global Distribution Matrix:")
print("-" * 40)
print(final_df["sentiment"].value_counts().to_string())
print("-" * 40)

# ==========================================
# PHASE 5: STRATIFIED TRAIN/TEST PARTITIONING
# ==========================================
print("\n[INFO] Executing rigorous 80/20 stratified matrix partitioning...")
# A custom mathematical stratified split (80/20) is implemented to ensure the target variable 
# probability distribution remains structurally identical across both training and testing sets.
train_parts = []
test_parts = []

for sentiment_name, group in final_df.groupby("sentiment"):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(len(group) * 0.8)

    train_group = group.iloc[:split_index].copy()
    test_group = group.iloc[split_index:].copy()

    train_parts.append(train_group)
    test_parts.append(test_group)

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"[SUCCESS] Dataset perfectly partitioned into independent subsets.")
print(f"|-- Training Partition Matrix Size   : {train_df.shape}")
print(f"|-- Evaluation Partition Matrix Size : {test_df.shape}")

print("\n[INFO] Training Matrix Stratification Profile:")
print("-" * 40)
print(train_df["sentiment"].value_counts().to_string())
print("-" * 40)

print("\n[INFO] Evaluation Matrix Stratification Profile:")
print("-" * 40)
print(test_df["sentiment"].value_counts().to_string())
print("-" * 40)

# ==========================================
# PHASE 6: DATA SERIALIZATION & EXECUTIVE SUMMARY
# ==========================================
print("\n[INFO] Serializing finalized partitions to disk...")
# The final, statistically sound curated splits are formally serialized to disk 
# for seamless ingestion by the TensorFlow sequential network.
final_df.to_csv(final_full_path, index=False)
train_df.to_csv(final_train_path, index=False)
test_df.to_csv(final_test_path, index=False)

print("\n" + "="*65)
print(" EXECUTIVE SUMMARY: FINALIZED NLP DATASETS")
print("="*65)
print(f"|-- Full Master Dataset Exported   : .\\{os.path.relpath(final_full_path, project_folder)}")
print(f"|-- Training Partition Exported    : .\\{os.path.relpath(final_train_path, project_folder)}")
print(f"|-- Evaluation Partition Exported  : .\\{os.path.relpath(final_test_path, project_folder)}")

print("\n[PREVIEW] Sanitized & Stratified Master Dataset (Sample View):")
print("-" * 65)
print(final_df.head(10))
print("-" * 65 + "\n")