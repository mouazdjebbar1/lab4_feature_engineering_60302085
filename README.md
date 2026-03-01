# lab4_feature_engineering_60302085

# 🧠 Azure ML Text Feature Engineering Pipeline  
**DSAI3202 – Lab 4**  
Student ID: 60302085  

---

## 📌 Overview

This lab implements a complete text feature engineering pipeline using Azure Machine Learning.

The objective of this lab was to:

- Build reusable Azure ML command components  
- Create a full feature engineering pipeline  
- Avoid data leakage  
- Generate structured features from text  
- Register the features inside Azure ML Feature Store  

The final output is a versioned, production-style feature set ready for modeling.

---

# 🏗️ Pipeline Architecture

The pipeline performs the following steps:
Gold Sampled Dataset
↓
Split (Train / Validation / Test)
↓
Text Normalization
↓
Feature Engineering
├── Review Length Features
├── Sentiment Features (VADER)
├── TF-IDF Features (500 terms)
├── SBERT Embeddings (384-dim)
↓
Merge All Features
↓
Feature Set Registration


All steps run entirely on Azure ML compute.

---

# 🔹 Components Implemented

## 1️⃣ Split Dataset Component

- Splits data into:
  - 70% Train
  - 15% Validation
  - 15% Test
- Uses fixed random seed
- Prevents data leakage

---

## 2️⃣ Text Normalization Component

Performs:

- Lowercasing
- URL removal
- Number replacement
- Punctuation removal
- Whitespace trimming
- Removes very short reviews (< 10 characters)

Ensures consistent preprocessing across all dataset splits.

---

## 3️⃣ Review Length Features

Extracted:

- `review_length_words`
- `review_length_chars`

These features capture verbosity and review engagement level.

---

## 4️⃣ Sentiment Features (VADER)

Extracted:

- `sentiment_pos`
- `sentiment_neg`
- `sentiment_neu`
- `sentiment_compound`

These features represent emotional tone and opinion polarity.

---

## 5️⃣ TF-IDF Features

- Fit only on the training split
- Applied to train, validation, and test
- Limited to 500 features (to reduce memory usage)
- Prevents vocabulary leakage

These features capture statistically important textual terms.

---

## 6️⃣ SBERT Embeddings

Model used:sentence-transformers/all-MiniLM-L6-v2

- 384-dimensional embeddings
- Context-aware sentence representation
- Captures semantic meaning of full review text

---

## 7️⃣ Merge Features Component

All engineered features are merged using:

- `asin`
- `reviewerID`

Memory optimization applied:

- Converted float64 → float32
- Progressive merging
- Garbage collection to reduce RAM usage

Final merged dataset:

- 914 total columns
- 911 engineered features

---

# ⚙️ Azure ML Pipeline Execution

Pipeline submitted using:

```bash
az ml job create -f pipelines/feature_pipeline.yml
TF-IDF features reduced from 5000 → 500 to resolve memory issues.

Final runtime:

~12 minutes


Project Structure:
components/
  split_dataset/
  normalize_text/
  review_length_features/
  vader_sentiment_features/
  tfidf_fit/
  tfidf_transform/
  sbert_embeddings/
  merge_features/

pipelines/
  feature_pipeline.yml

feature_store/
  FeatureSetSpec.yaml
  feature_set_amazon_review_text_features.yml
databricksNoteBook
  assigment1_featuresV1.ipynb

README.md
