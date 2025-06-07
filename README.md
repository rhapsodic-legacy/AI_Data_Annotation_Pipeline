# AI Data Annotation Pipeline
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

