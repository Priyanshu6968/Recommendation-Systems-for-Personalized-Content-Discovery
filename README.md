# Recommendation Systems for Personalized Content Discovery

## Overview
This repository contains a production-ready recommendation system built to handle highly sparse implicit and explicit feedback datasets. The system was developed through a structured 12-phase engineering lifecycle, resulting in a Hybrid Ensemble architecture that prioritizes both predictive accuracy and interpretability.

## Architecture
The core architecture fuses multiple models to capture different signals in user behavior:

1. **Global Baseline**: Captures overarching popularity and user/item biases.
2. **Item-Based Collaborative Filtering (KNN)**: Captures localized item-to-item similarity patterns.
3. **Funk SVD (Matrix Factorization)**: Captures latent dimensions and hidden user-item interactions.
4. **Hybrid Ensemble**: Blends the latent feature detection of SVD (70% weight) with the neighborhood explainability of Item-CF (30% weight) to minimize uncorrelated error types.
5. **Explainability Engine (XAI)**: Demystifies recommendations by translating mathematical item-item correlations into natural language justifications for the end user.

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Pip package manager

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the raw dataset files (e.g., `movie_titles.csv`) are located in the `dataset/` directory.

### Running the System
- **Data Pipeline**: Run the parser to convert text files to compressed Parquet format.
  ```bash
  python scripts/run_parser.py
  ```
- **Evaluation Framework**: Generate a comprehensive scorecard for the models.
  ```bash
  python scripts/run_evaluation.py
  ```
- **Streamlit Dashboard**: Launch the interactive local web application to view recommendations and explanations.
  ```bash
  streamlit run app.py
  ```

## Evaluation Results
The system is evaluated on both error magnitude and top-K ranking relevance.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **RMSE** | 0.8049 | Root Mean Square Error (Ensemble model) |
| **MAP@10** | 0.8569 | Mean Average Precision at rank 10 |
| **Precision@10** | 0.8849 | Percentage of relevant items in the top 10 |
| **Recall@10** | 0.2272 | Percentage of total relevant items captured |
| **NDCG@10** | 0.9014 | Normalized Discounted Cumulative Gain |
| **Hit Rate** | 0.9980 | Ratio of users receiving at least one relevant hit |
| **Coverage** | 0.9460 | Percentage of catalog recommended across all users |

## Explainable AI
The built-in explanation engine generates post-hoc justifications for recommendations. Example output:
"We recommend 'Movie A' because you previously enjoyed 'Movie B' (Similarity Score: 0.89)."