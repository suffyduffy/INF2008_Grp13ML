import numpy as np
import matplotlib.pyplot as plt

# Define model names
models = ["Random Forest", "Neural Network", "SVR"]

# Define the updated metrics
r2_scores = [0.8075, 0.8431, 0.4909]  # R² Score (Higher is better)
mae_scores = [56430.30, 46115.61, 76140.28]  # Mean Absolute Error (Lower is better)
mse_scores = [5348514486.05, 4048307830.83, 12210140226.56]  # Mean Squared Error (Lower is better)

# Set colors for each model
colors = ['green', 'blue', 'orange']

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)  # Compact layout

# Function to add labels above the bars
def add_labels(ax, values):
    for i, v in enumerate(values):
        formatted_value = f"{v:,.0f}" if v >= 1e6 else f"{v:,.2f}"  # Format large numbers
        ax.text(i, v * 1.05, formatted_value, ha='center', fontsize=8, fontweight='bold', color='black')

# R² Score (Higher is better)
axes[0].bar(models, r2_scores, color=colors)
axes[0].set_title("R² Score Comparison", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Higher is Better", fontsize=8)
axes[0].set_ylim(0, 1.1)  # More space at the top
add_labels(axes[0], r2_scores)

# MAE (Lower is better)
axes[1].bar(models, mae_scores, color=colors)
axes[1].set_title("Mean Absolute Error (MAE)", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Lower is Better", fontsize=8)
axes[1].set_ylim(0, max(mae_scores) * 1.3)  # Increased ylim for spacing
add_labels(axes[1], mae_scores)

# MSE (Lower is better)
axes[2].bar(models, mse_scores, color=colors)
axes[2].set_title("Mean Squared Error (MSE)", fontsize=10, fontweight="bold")
axes[2].set_ylabel("Lower is Better", fontsize=8)
axes[2].set_ylim(0, max(mse_scores) * 1.3)  # Increased ylim for spacing
add_labels(axes[2], mse_scores)

# Show the plot
plt.show()
