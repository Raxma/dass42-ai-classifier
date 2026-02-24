# preprocess/preprocess.py

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------
# Config
# --------------------------
RAW_DATA_PATH = Path('C:/Users/rasman khurshid/Downloads/data.csv')  # Update this path if needed
CLEANED_DATA_PATH = Path('C:/Users/rasman khurshid/Downloads/cleaned_dass_data.csv')
CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------
# Load Data
# --------------------------
df = pd.read_csv(RAW_DATA_PATH, delimiter="\t")
q_columns = [f"Q{i}A" for i in range(1, 43)]
dass_df = df[q_columns].copy()

# --------------------------
# Define Subsets for Scores
# --------------------------
depression_cols = [f"Q{i}A" for i in [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]]
anxiety_cols =    [f"Q{i}A" for i in [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]]
stress_cols =     [f"Q{i}A" for i in [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]]

# --------------------------
# Scoring
# --------------------------
dass_df["Depression_Score"] = dass_df[depression_cols].sum(axis=1)
dass_df["Anxiety_Score"] = dass_df[anxiety_cols].sum(axis=1)
dass_df["Stress_Score"] = dass_df[stress_cols].sum(axis=1)

# --------------------------
# Labeling Functions
# --------------------------
def label_depression(score):
    return (
        "Normal" if score <= 9 else
        "Mild" if score <= 13 else
        "Moderate" if score <= 20 else
        "Severe" if score <= 27 else
        "Extremely Severe"
    )

def label_anxiety(score):
    return (
        "Normal" if score <= 7 else
        "Mild" if score <= 9 else
        "Moderate" if score <= 14 else
        "Severe" if score <= 19 else
        "Extremely Severe"
    )

def label_stress(score):
    return (
        "Normal" if score <= 14 else
        "Mild" if score <= 18 else
        "Moderate" if score <= 25 else
        "Severe" if score <= 33 else
        "Extremely Severe"
    )

# --------------------------
# Apply Labels
# --------------------------
dass_df["Depression_Label"] = dass_df["Depression_Score"].apply(label_depression)
dass_df["Anxiety_Label"] = dass_df["Anxiety_Score"].apply(label_anxiety)
dass_df["Stress_Label"] = dass_df["Stress_Score"].apply(label_stress)

# --------------------------
# Basic Validation
# --------------------------
dass_df.drop_duplicates(inplace=True)
assert dass_df.isnull().sum().sum() == 0, "Null values found!"

# --------------------------
# Label Validation
# --------------------------
def validate_label(column, label_func):
    mismatches = dass_df.assign(_check=dass_df[column].apply(label_func))
    return (mismatches["_check"] != dass_df[column.replace("_Label", "_Label")]).sum()

print("Depression mismatches:", validate_label("Depression_Label", label_depression))
print("Anxiety mismatches:", validate_label("Anxiety_Label", label_anxiety))
print("Stress mismatches:", validate_label("Stress_Label", label_stress))

# --------------------------
# Save Cleaned Data
# --------------------------
dass_df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"✅ Cleaned data saved to {CLEANED_DATA_PATH}")

# --------------------------
# Optional: Class Distribution Plot
# --------------------------
def plot_class_distribution():
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.countplot(ax=axes[0], data=dass_df, x="Depression_Label",
                  order=["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"])
    axes[0].set_title("Depression Severity Distribution")

    sns.countplot(ax=axes[1], data=dass_df, x="Anxiety_Label",
                  order=["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"])
    axes[1].set_title("Anxiety Severity Distribution")

    sns.countplot(ax=axes[2], data=dass_df, x="Stress_Label",
                  order=["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"])
    axes[2].set_title("Stress Severity Distribution")

    plt.tight_layout()
    plt.show()

# Call this function when running manually
if __name__ == "__main__":
    plot_class_distribution()
    
def load_and_prepare_data(csv_path):
    import pandas as pd

    # Load the raw CSV
    df = pd.read_csv(csv_path, delimiter='\t')

    # Select Q1A to Q42A
    q_columns = [f"Q{i}A" for i in range(1, 43)]
    dass_df = df[q_columns].copy()

    # Compute scores
    depression_cols = [f"Q{i}A" for i in [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]]
    anxiety_cols = [f"Q{i}A" for i in [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]]
    stress_cols = [f"Q{i}A" for i in [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]]

    dass_df["Depression_Score"] = dass_df[depression_cols].sum(axis=1)
    dass_df["Anxiety_Score"] = dass_df[anxiety_cols].sum(axis=1)
    dass_df["Stress_Score"] = dass_df[stress_cols].sum(axis=1)

    # Label functions
    def label(score, bounds):
        for upper, label in bounds:
            if score <= upper:
                return label
        return bounds[-1][1]

    depression_bounds = [(9, "Normal"), (13, "Mild"), (20, "Moderate"), (27, "Severe"), (float('inf'), "Extremely Severe")]
    anxiety_bounds = [(7, "Normal"), (9, "Mild"), (14, "Moderate"), (19, "Severe"), (float('inf'), "Extremely Severe")]
    stress_bounds = [(14, "Normal"), (18, "Mild"), (25, "Moderate"), (33, "Severe"), (float('inf'), "Extremely Severe")]

    dass_df["Depression_Label"] = dass_df["Depression_Score"].apply(lambda x: label(x, depression_bounds))
    dass_df["Anxiety_Label"] = dass_df["Anxiety_Score"].apply(lambda x: label(x, anxiety_bounds))
    dass_df["Stress_Label"] = dass_df["Stress_Score"].apply(lambda x: label(x, stress_bounds))

    # Drop duplicates
    dass_df = dass_df.drop_duplicates()

    return dass_df

