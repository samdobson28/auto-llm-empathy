#!/usr/bin/env python3
"""
analyze.py

This script analyzes the LLM empathy ratings by comparing them to the ground truth.
It assumes the input CSV (merged_results.csv) contains:
  - Ground truth category in the "Category" column (values like "empathetic", "neutral", "anti").
  - LLM numeric ratings in columns: "transcript_empathy_rating", "audio_features_empathy_rating", and "combined_empathy_rating".

It performs two analyses:
  1. Computes the mean absolute error (MAE) between the LLM’s numeric rating and the ground truth (mapped to numeric values: anti=1, neutral=3, empathetic=5).
  2. Maps the numeric LLM ratings to categorical labels (rating <=2 => "anti", rating == 3 => "neutral", rating >=4 => "empathetic") and prints a classification report and confusion matrix.

The full results (including computed error columns) are saved to "analysis_results.csv".
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Specify the input CSV file (ensure this file exists in your working directory)
filename = "classification_results.csv"  # Change if needed

# Load the data
df = pd.read_csv(filename)

# Mapping for ground truth categorical labels to numeric scores
truth_mapping = {"anti": 1, "neutral": 3, "empathetic": 5}

# Standardize and map the ground truth Category column
df["Category_clean"] = df["Category"].astype(str).str.lower().str.strip()
df["truth_numeric"] = df["Category_clean"].map(truth_mapping)

# Define the LLM rating columns from auto-classify.py
rating_columns = ['transcript_empathy_rating', 'audio_features_empathy_rating', 'combined_empathy_rating']

# Convert rating columns to numeric (if not already)
for col in rating_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute absolute error for each rating column against the ground truth numeric score
for col in rating_columns:
    error_col = col + "_abs_error"
    df[error_col] = (df[col] - df["truth_numeric"]).abs()

# Print mean absolute errors for each method
for col in rating_columns:
    error_col = col + "_abs_error"
    mean_error = df[error_col].mean()
    print(f"Mean absolute error for {col}: {mean_error:.2f}")

# Define a function to map a numeric rating to a categorical label
def numeric_to_category(rating):
    try:
        rating = float(rating)
        if rating <= 2:
            return "anti"
        elif rating == 3:
            return "neutral"
        elif rating >= 4:
            return "empathetic"
    except Exception:
        return np.nan

# Create predicted categorical columns from the LLM numeric ratings
for col in rating_columns:
    pred_col = col + "_predicted"
    df[pred_col] = df[col].apply(numeric_to_category)

# Evaluate classification performance for each method
methods = [col + "_predicted" for col in rating_columns]

for method in methods:
    print(f"\n--- Classification report for {method} ---")
    # Fill missing values with a placeholder (if any)
    preds = df[method].fillna("Unknown")
    truth = df["Category_clean"].fillna("Unknown")
    
    print("Classification Report:")
    print(classification_report(truth, preds, labels=["empathetic", "neutral", "anti"], zero_division=0))
    
    cm = confusion_matrix(truth, preds, labels=["empathetic", "neutral", "anti"])
    print("Confusion Matrix:")
    print(cm)

# Save the full analysis DataFrame to a CSV file
output_filename = "analysis_results.csv"
df.to_csv(output_filename, index=False)
print(f"\nAnalysis results saved to {output_filename}.")
