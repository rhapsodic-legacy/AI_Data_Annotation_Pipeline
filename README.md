# AI Data Annotation Pipeline

## Overview
This repository contains an AI Data Annotation Pipeline designed to preprocess, annotate, and evaluate text data for sentiment analysis. The pipeline processes movie reviews from the IMDb dataset, using DistilBERT to predict sentiment (positive or negative), refines low-confidence predictions through prompt engineering, and evaluates performance with detailed metrics and visualizations. Its purpose is to demonstrate end-to-end data annotation workflows, showcasing skills in natural language processing (NLP), data ethics, and model evaluation, applicable to AI training and research.

Key features:
- **Preprocessing**: Removes personally identifiable information (PII) using spaCy and Regex for ethical data handling.
- **Annotation**: Applies DistilBERT for sentiment classification.
- **Prompt Engineering**: Refines predictions with confidence below 0.95 to improve accuracy.
- **Evaluation**: Computes class-specific and weighted precision, recall, and F1-score.
- **Visualization**: Generates confidence histograms and confusion matrices.
- **Error Analysis**: Identifies patterns in misclassified samples.

## Pipeline Diagram
The following Mermaid diagram illustrates the pipeline’s workflow:

```mermaid
graph TD
    A[Raw IMDb Data] --> B[Preprocess PII Removal]
    B --> C[DistilBERT Annotation]
    C --> D[Refine Prompt Engineering]
    D --> E[Evaluate Metrics]
    E --> F[Visualize Histogram Confusion Matrix]
    F --> G[Analyze Error Patterns]
    G --> H[Save Outputs CSV PNGs]

This pipeline demonstrates end-to-end data annotation and model evaluation for AI training, built for the Outlier Computer Programming AI Trainer role. It runs in Google Colab (free tier) and includes:

- **Data Preprocessing**: Cleans text and removes PII using spaCy and Regex, inspired by my work on the Hugging Face BLOOM LLM PII toolkit.
- **Model Annotation**: Annotates sentiment analysis outputs from DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`).
- **Prompt Engineering**: Refines prompts to improve low-confidence predictions, simulating AI training tasks.
- **Evaluation**: Computes precision, recall, F1-score, and visualizes results (confidence histograms, confusion matrix).

## Setup
1. Run in Google Colab.
2. Install dependencies: `transformers`, `pandas`, `spacy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `datasets`, `fsspec`, `torchvision`.
3. Use the IMDb dataset (public, Hugging Face).

## Outputs
- `annotated_imdb_outputs.csv`: Annotated dataset with model predictions and metrics.
- `confidence_histogram.png`: Distribution of confidence scores.
- `confusion_matrix.png`: Confusion matrix for refined predictions.

## Sample Results (First 1000 IMDb Test Samples)
- **Original Model**: Precision=1.00, Recall=0.93, F1=0.96
- **Refined Model**: Precision=1.00, Recall=0.93, F1=0.96
- **Note**: Metrics use weighted averages. Precision appears perfect (likely rounded), though a false positive exists in a small sample, suggesting rarity in the full dataset. Prompt refinement had no impact here, possibly due to high initial confidence.

