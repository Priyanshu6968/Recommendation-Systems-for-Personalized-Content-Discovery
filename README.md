# Netflix Prize Recommendation System

This repository contains a state-of-the-art recommendation engine designed to tackle the Netflix Prize Dataset. It was built with modern MLOps practices to maximize recommendation quality, RMSE, and MAP@10 scores.

## Project Structure

- `data/`: Contains raw, interim, and processed data (ignored by git).
- `src/`: Source code including data processing, feature engineering, models, and evaluation.
- `notebooks/`: Jupyter notebooks for EDA and prototyping.
- `configs/`: YAML configuration files.
- `scripts/`: Entry point scripts for training and execution.

## Getting Started

1. Create a virtual environment and install dependencies:
```bash
pip install -e .[dev]
```
2. Place the Netflix Prize data files in `data/raw/`.
3. Track experiments with Weights & Biases.