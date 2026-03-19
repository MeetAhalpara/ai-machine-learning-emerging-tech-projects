"""
Sentiment Corpus Generation Pipeline
Author: Meet Ahalpara

This module operates as the foundational data engineering pipeline for a 4-class 
Natural Language Processing (NLP) sentiment classifier. It ingests an authentic 
IMDB dataset, sanitizes raw textual features, and programmatically generates 
synthetic positive, negative, mixed, and neutral reviews to ensure absolute 
class equilibrium prior to model training.
"""
import random
import re
import os
import pandas as pd
# from datasets import load_dataset   # Keep this for future Hugging Face use

random.seed(42)

print("\n" + "="*65)
print(" NLP SENTIMENT CORPUS GENERATION PIPELINE INITIATED")
print("="*65)

# ==========================================
# PHASE 1: ENVIRONMENT & PATH RESOLUTION
# ==========================================
# The absolute path of the current script's directory is dynamically resolved.
# This ensures the pipeline executes seamlessly regardless of the local environment.
project_folder = os.path.dirname(os.path.abspath(__file__))
dataset_root_folder = os.path.join(project_folder, "Dataset")
dataset_folder = os.path.join(dataset_root_folder, "ReviewDatasets")

csv_dataset_path = os.path.join(dataset_root_folder, "IMDB Dataset.csv")

os.makedirs(dataset_folder, exist_ok=True)

if not os.path.exists(csv_dataset_path):
    raise FileNotFoundError(f"IMDB dataset file not found: {csv_dataset_path}")

# ==========================================
# PHASE 2: DATA INGESTION
# ==========================================
print("\n[INFO] Ingesting foundational IMDB dataset...")

# ---- Primary Protocol: Load from local CSV file ----
df = pd.read_csv(csv_dataset_path)

# ---- Alternative Protocol: Load from Hugging Face ----
# dataset = load_dataset("Omarrran/50k_IMBD_Movie_Review_by_HNM")
# df = pd.DataFrame(dataset["train"])

print(f"[SUCCESS] Dataset successfully loaded. Shape: {df.shape}")
print(f"|-- Detected Data Columns: {df.columns.tolist()}")
print("\n[PREVIEW] Foundational Data Structure (First 5 Records):")
print("-" * 65)
print(df.head())
print("-" * 65)

# ==========================================
# PHASE 3: SCHEMA DETECTION & STANDARDIZATION
# ==========================================
print("\n[INFO] Executing heuristic schema detection...")
# The pipeline employs heuristic scanning to dynamically identify feature and target 
# columns, ensuring compatibility across various dataset schemas.
possible_text_columns = ["text", "review", "content", "sentence"]
text_col = None

for col in possible_text_columns:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    raise ValueError("No suitable text column found. Check dataset columns.")

# Dynamically identify the target/label column mapping.
possible_label_columns = ["label", "sentiment", "target", "class"]
label_col = None

for col in possible_label_columns:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError("No suitable label column found. Check dataset columns.")

print(f"[SUCCESS] Schema mapped successfully.")
print(f"|-- Selected Feature Matrix Column : '{text_col}'")
print(f"|-- Selected Target Vector Column  : '{label_col}'")

print("\n[INFO] Initializing text sanitization protocols...")

def clean_text(text):
    """
    Standardizes raw text arrays to ensure mathematical uniformity during tokenization.
    
    Args:
        text (str/object): The raw textual review.
        
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
print("[SUCCESS] Corpus sanitization complete.")

# ==========================================
# PHASE 4: BASE DISTRIBUTION ANALYSIS
# ==========================================
print("\n[INFO] Analyzing class distribution across the foundational corpus...")
print("-" * 40)
print(df[label_col].value_counts().to_string()) 
print("-" * 40)

print("\n[PREVIEW] Sanitized Feature Space (Sample View):")
print("-" * 65)
print(df[[text_col, label_col]].head(5)) 
print("-" * 65)

# Programmatically determine label encoding configurations (binary vs. string representations)
# to strictly partition the dataset into isolated positive and negative matrices.
unique_labels = sorted(df[label_col].dropna().unique().tolist())

positive_df = None
negative_df = None

if set(unique_labels) == {0, 1}:
    positive_df = df[df[label_col] == 1].copy()
    negative_df = df[df[label_col] == 0].copy()

elif set(str(x).lower() for x in unique_labels) == {"positive", "negative"}:
    positive_df = df[df[label_col].astype(str).str.lower() == "positive"].copy()
    negative_df = df[df[label_col].astype(str).str.lower() == "negative"].copy()

else:
    raise ValueError(
        "Could not automatically detect positive/negative labels. "
        "Please print the label column and set them manually."
    )

print(f"|-- Isolated Positive Instances : {len(positive_df)}")
print(f"|-- Isolated Negative Instances : {len(negative_df)}")

# ==========================================
# PHASE 5: SYNTHETIC CORPUS GENERATION
# ==========================================
print("\n[INFO] Initializing synthetic data generation protocols...")
# Pre-defined lexicons capturing absolute positive and negative sentiment signals.
# These arrays serve as the architectural foundation for generating targeted, 
# mathematically polarized synthetic training samples.
positive_manual_phrases = [
    "the acting was amazing",
    "the visuals were beautiful",
    "the soundtrack was excellent",
    "the performances were strong",
    "the movie was emotional",
    "the story was engaging",
    "the direction was impressive",
    "the cinematography was stunning",
    "the characters were memorable",
    "the film had a powerful message",
    "the movie was enjoyable",
    "the dialogue was meaningful",
    "the cast was excellent",
    "the ending was satisfying",
    "the plot was interesting",
    "the film was inspiring",
    "the screenplay was thoughtful",
    "the scenes were well crafted",
    "the movie was entertaining",
    "the story was beautifully written"
]

negative_manual_phrases = [
    "the story was weak",
    "the pacing was slow",
    "the ending was disappointing",
    "the acting felt forced",
    "the movie was boring",
    "the plot was confusing",
    "the dialogue was awkward",
    "the film was dull",
    "the script was poor",
    "the characters were unconvincing",
    "the direction felt messy",
    "the movie was predictable",
    "the scenes felt too long",
    "the film made little sense",
    "the experience was frustrating",
    "the screenplay was weak",
    "the movie was forgettable",
    "the story was poorly written",
    "the performances felt flat",
    "the film was badly executed"
]

positive_phrase_bank = positive_manual_phrases
negative_phrase_bank = negative_manual_phrases

# Generation Array 1: Positive Short-Form Reviews
# This augments the dataset with concise positive text, accelerating the model's 
# ability to learn explicit, high-density sentiment markers.
positive_templates = [
    "{phrase}.",
    "overall, {phrase}.",
    "in my opinion, {phrase}.",
    "throughout the movie, {phrase}.",
    "from the beginning, {phrase}.",
    "for me, {phrase}."
]

positive_reviews = []

for _ in range(2000):
    phrase = random.choice(positive_phrase_bank)
    template = random.choice(positive_templates)

    positive_sentence = template.format(phrase=phrase).strip().capitalize()
    positive_reviews.append(positive_sentence)

positive_short_df = pd.DataFrame({
    "text": positive_reviews,
    "sentiment": ["positive"] * len(positive_reviews)
})

# Generation Array 2: Negative Short-Form Reviews
# Augments the dataset with concise negative text to guarantee strict class balance 
# and improve algorithmic generalization on heavily polarized sub-sequences.
negative_templates = [
    "{phrase}.",
    "overall, {phrase}.",
    "in my opinion, {phrase}.",
    "throughout the movie, {phrase}.",
    "for me, {phrase}.",
    "by the end, {phrase}."
]

negative_reviews = []

for _ in range(2000):
    phrase = random.choice(negative_phrase_bank)
    template = random.choice(negative_templates)

    negative_sentence = template.format(phrase=phrase).strip().capitalize()
    negative_reviews.append(negative_sentence)

negative_short_df = pd.DataFrame({
    "text": negative_reviews,
    "sentiment": ["negative"] * len(negative_reviews)
})

# Generation Array 3: Mixed Sentiment Reviews
# Synthetic 'mixed' reviews are programmatically generated by bridging polarized 
# sentiment phrases with contrasting conjunctions, essential for the 4-class architecture.
connectors = ["but", "however", "although", "yet"]

mixed_reviews = []

for _ in range(2000):
    pos_phrase = random.choice(positive_phrase_bank)
    neg_phrase = random.choice(negative_phrase_bank)
    connector = random.choice(connectors)

    mixed_sentence = f"{pos_phrase} {connector} {neg_phrase}."
    mixed_sentence = mixed_sentence.capitalize()

    mixed_reviews.append(mixed_sentence)

mixed_df = pd.DataFrame({
    "text": mixed_reviews,
    "sentiment": ["mixed"] * len(mixed_reviews)
})

# Generation Array 4: Neutral Informational Reviews
# Neutral synthetic subsets focus on objective cinematic facts rather than subjective opinions.
# This mathematically trains the network to isolate informational text from emotional density.
subjects = [
    "the movie",
    "the film",
    "this movie",
    "this film",
    "the story",
    "the plot"
]

neutral_facts = [
    "is two hours long",
    "was released in 2020",
    "belongs to the drama genre",
    "has several main characters",
    "contains multiple dialogue scenes",
    "is set in a city environment",
    "includes action sequences",
    "has a runtime close to two hours",
    "was directed by a known director",
    "was discussed by viewers online",
    "follows a family based storyline",
    "takes place over several days",
    "features a small main cast",
    "is available in multiple languages",
    "includes indoor and outdoor scenes"
]

neutral_templates = [
    "{subject} {fact}.",
    "according to the review, {subject} {fact}.",
    "{subject} {fact} in the story.",
    "overall, {subject} {fact}.",
    "{subject} {fact} for most of the runtime."
]

neutral_reviews = []

for _ in range(2000):
    subject = random.choice(subjects)
    fact = random.choice(neutral_facts)
    template = random.choice(neutral_templates)

    neutral_sentence = template.format(subject=subject, fact=fact).strip().capitalize()
    neutral_reviews.append(neutral_sentence)

neutral_df = pd.DataFrame({
    "text": neutral_reviews,
    "sentiment": ["neutral"] * len(neutral_reviews)
})

print("[SUCCESS] Synthetic corpora generation successfully completed (8000 target samples).")

# ==========================================
# PHASE 6: DATA SERIALIZATION
# ==========================================
print("\n[INFO] Serializing augmented datasets to disk...")

# The fully augmented datasets are formally serialized to the local environment, 
# preparing them for immediate downstream consumption by the stratification pipeline.
positive_short_path = os.path.join(dataset_folder, "positive_short_reviews.csv")
negative_short_path = os.path.join(dataset_folder, "negative_short_reviews.csv")
mixed_path = os.path.join(dataset_folder, "mixed_reviews.csv")
neutral_path = os.path.join(dataset_folder, "neutral_reviews.csv")

positive_short_df.to_csv(positive_short_path, index=False)
negative_short_df.to_csv(negative_short_path, index=False)
mixed_df.to_csv(mixed_path, index=False)
neutral_df.to_csv(neutral_path, index=False)

print("[SUCCESS] Serialization complete.")
print(f"|-- Exported: {positive_short_path}")
print(f"|-- Exported: {negative_short_path}")
print(f"|-- Exported: {mixed_path}")
print(f"|-- Exported: {neutral_path}")

# ==========================================
# PHASE 7: EXECUTIVE PREVIEW
# ==========================================
print("\n" + "="*65)
print(" EXECUTIVE SUMMARY: GENERATED SYNTHETIC DATASETS")
print("="*65)

print(f"|-- Positive Matrix Vector : {positive_short_df.shape}")
print(f"|-- Negative Matrix Vector : {negative_short_df.shape}")
print(f"|-- Mixed Matrix Vector    : {mixed_df.shape}")
print(f"|-- Neutral Matrix Vector  : {neutral_df.shape}")

print("\n[PREVIEW] Sample Positive Short Reviews:")
print(positive_short_df.head(5))

print("\n[PREVIEW] Sample Negative Short Reviews:")
print(negative_short_df.head(5))

print("\n[PREVIEW] Sample Mixed Reviews:")
print(mixed_df.head(5))

print("\n[PREVIEW] Sample Neutral Reviews:")
print(neutral_df.head(5))
print("\n" + "="*65 + "\n")