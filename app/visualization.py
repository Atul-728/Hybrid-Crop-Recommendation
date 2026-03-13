import os

# Force non-GUI backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join("data", "crop_recommendation.csv")
CHART_PATH = os.path.join("static", "class_distribution.png")


def generate_class_distribution():

    df = pd.read_csv(DATA_PATH)

    # Count classes and sort descending
    counts = df["label"].value_counts().sort_values(ascending=False)

    # Modern style
    sns.set_style("whitegrid")

    plt.figure(figsize=(14, 7))

    # Color palette
    colors = sns.color_palette("viridis", len(counts))

    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        palette=colors
    )

    # Title & labels
    plt.title("Crop Dataset Class Distribution", fontsize=18, weight="bold")
    plt.xlabel("Crop Type", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)

    plt.xticks(rotation=60, ha="right")

    # Add value labels on bars
    for i, value in enumerate(counts.values):
        ax.text(i, value + 1, str(value), ha='center', fontsize=10)

    plt.tight_layout()

    # Save high quality image
    plt.savefig(CHART_PATH, dpi=300, bbox_inches="tight")
    plt.close()