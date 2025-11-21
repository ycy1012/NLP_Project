# Rating Prediction from Yelp Reviews Using Transformer Models

**Authors:** Binqian Chai, Zhihao Chen, Chenyao Yu  
**Course:** ECE 684: Natural Language Processing  
**Date:** November 2025

## Project Overview
This project investigates explainable rating prediction on the Yelp Review Full dataset. We compare two problem formulations—**Multi-class Classification** and **Ordinal Regression**—across three different model architectures:
1.  **Baseline:** TF-IDF + Logistic Regression
2.  **DistilBERT:** Fine-tuned for classification and regression
3.  **RoBERTa:** Fine-tuned for classification and regression

## System Requirements

### Python & Hardware
* **Python 3.8+** is required.
* **Hardware:** The **Baseline** model runs on a standard CPU. The **Transformer models** (DistilBERT/RoBERTa) require a GPU (e.g., NVIDIA RTX 5000 or A100) for efficient training.

### Libraries
This project relies on the Hugging Face ecosystem and PyTorch. Install all dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch transformers datasets evaluate accelerate
````

## Dataset

We use the **Yelp Review Full** dataset.

  * **Source:** [Hugging Face Datasets](https://www.google.com/search?q=https://huggingface.co/datasets/yelp_review_full)
  * **Setup:** You do **not** need to download a CSV file manually. The scripts automatically download and cache the data via the `load_dataset('yelp_review_full')` command.
