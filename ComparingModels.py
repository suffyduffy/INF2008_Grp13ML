import numpy as np
import matplotlib.pyplot as plt

# Define model names
models = ["Random Forest", "Neural Network", "SVM"]

# Define the metrics
r2_scores = [0.8075, 0.9064, 0.4909]  # R² Score (Higher is Better)
mae_scores = [56430, 37409, 76140]  # Mean Absolute Error (Lower is Better)
mse_scores = [5.3e9, 2.5e9, 1.2e10]  # Mean Squared Error (Lower is Better)

# Set colors for each model
colors = ['green', 'blue', 'orange']

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # Increased width for better spacing

# Function to add labels **above** the bars
def add_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v * 1.05, f"{v:,.4g}" if v >= 1e6 else f"{v:,}",
                ha='center', fontsize=12, fontweight='bold', color='black')

# R² Score (Higher is better)
axes[0].bar(models, r2_scores, color=colors)
axes[0].set_title("R² Score Comparison", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Higher is Better", fontsize=12)
axes[0].set_ylim(0, 1.1)  # More space at the top
add_labels(axes[0], r2_scores)

# MAE (Lower is better)
axes[1].bar(models, mae_scores, color=colors)
axes[1].set_title("Mean Absolute Error (MAE)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Lower is Better", fontsize=12)
axes[1].set_ylim(0, max(mae_scores) * 1.3)  # Increased ylim for spacing
add_labels(axes[1], mae_scores)

# MSE (Lower is better)
axes[2].bar(models, mse_scores, color=colors)
axes[2].set_title("Mean Squared Error (MSE)", fontsize=14, fontweight="bold")
axes[2].set_ylabel("Lower is Better", fontsize=12)
axes[2].set_ylim(0, max(mse_scores) * 1.3)  # Increased ylim for spacing
add_labels(axes[2], mse_scores)

# Improve layout
plt.tight_layout(pad=2)  # Add spacing between charts
plt.show()
