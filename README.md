# Automated Customer Review Analysis

An NLP pipeline that classifies Amazon product review sentiment, clusters products into meta-categories, and generates blog-style recommendation articles using generative AI.

## Project Overview

With thousands of reviews spread across multiple platforms, manually analyzing customer feedback is impractical. This project automates the process through three stages:

1. **Sentiment Classification** - Classify reviews as positive, negative, or neutral using a pre-trained RoBERTa transformer model.
2. **Product Category Clustering** - Group products into 6 meta-categories using sentence embeddings and K-Means clustering, refined by an LLM.
3. **Blog Post Generation** - Summarize reviews into recommendation articles for each category using a local LLM (Ollama / Qwen 2.5).

## Dataset

- **Source**: [Datafiniti Amazon Consumer Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data)
- **Size**: 28,332 reviews across 24 columns (ratings, review text, product info, categories, etc.)
- **Balanced subset**: 3,618 reviews (1,206 per sentiment class) used for classification evaluation

## Pipeline

```
Raw Reviews (28,332)
    |
    v
Data Cleaning & Preprocessing
    |
    v
Sentiment Labeling (star ratings -> negative / neutral / positive)
    |
    v
Class Balancing (downsample to 1,206 per class)
    |                                       |
    v                                       v
Sentiment Classification             Category Clustering
(CardiffNLP RoBERTa)                 (SentenceTransformer + KMeans + Ollama)
    |                                       |
    v                                       v
Classification Metrics               6 Meta-Categories
                                            |
                                            v
                                    Blog Post Generation
                                    (Ollama / Qwen 2.5)
                                            |
                                            v
                                    summaries.json
```

## Results

### Sentiment Classification

Model: `cardiffnlp/twitter-roberta-base-sentiment`

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Negative | 0.68      | 0.76   | 0.72     |
| Neutral  | 0.53      | 0.26   | 0.35     |
| Positive | 0.65      | 0.91   | 0.76     |
| **Overall Accuracy** | | | **64%** |

### Product Categories

Products were clustered into 6 meta-categories:

| Category | Description |
|----------|-------------|
| Health & Beauty | Batteries, personal care, health items |
| Electronics | Audio, video, smart home, cameras |
| Tablets & E-readers | Fire tablets, Kindle devices |
| Home & Kitchen | Kitchen appliances, storage, accessories |
| Office Supplies | Laptop stands, desk accessories |
| Pet Supplies | Dog crates, cat litter boxes, pet accessories |

### Generated Blog Posts

The pipeline produces category-specific blog articles covering:
- Top 3 products with average ratings and review counts
- Key features, pros, and cons
- Top complaints from reviews
- Worst product in each category and reasons to avoid it

Output is saved to `notebooks/summaries.json`.

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Sentiment model | `cardiffnlp/twitter-roberta-base-sentiment` (Hugging Face Transformers) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Clustering | scikit-learn KMeans |
| Category classification & blog generation | Ollama (Qwen 2.5) |
| Evaluation | scikit-learn (classification_report, confusion_matrix) |

## Project Structure

```
.
├── README.md
├── data/
│   └── amazon-customer-reviews/       # Raw CSV files from Kaggle
├── notebooks/
│   ├── main.ipynb                     # Full pipeline notebook
│   ├── Amazon_Reviews_Fine_Tuning_Roberta_Base.ipynb  # Fine-tuning experiment
│   └── summaries.json                 # Generated blog post output
└── .gitignore
```

## Setup & Usage

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed locally with the Qwen 2.5 model pulled

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn transformers sentence-transformers ollama
```

### Running

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data) and place the CSV files in `data/amazon-customer-reviews/`.
2. Pull the Qwen 2.5 model for Ollama:
   ```bash
   ollama pull qwen2.5
   ```
3. Open and run `notebooks/main.ipynb` from top to bottom.
4. Generated blog summaries will be saved to `notebooks/summaries.json`.

## Notebooks

- **main.ipynb** - End-to-end pipeline: data loading, preprocessing, sentiment classification, category clustering, and blog post generation.
- **Amazon_Reviews_Fine_Tuning_Roberta_Base.ipynb** - Experimental notebook for fine-tuning a RoBERTa model on the balanced review dataset as an alternative to the pre-trained classifier.
