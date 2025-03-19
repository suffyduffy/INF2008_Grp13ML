import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "cleanedHDB.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Convert 'month' column to datetime format if it exists
if 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
else:
    print("Error: 'month' column is missing from the dataset.")

# Collate data for years 2017 to 2025
df_filtered = df[(df['year'] >= 2017) & (df['year'] <= 2025)]

# Calculate average resale price per year
avg_prices = df_filtered.groupby('year')['resale_price'].mean()

# Plot the line chart
plt.figure(figsize=(10, 5))
plt.plot(avg_prices.index, avg_prices.values, marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xlabel("Year")
plt.ylabel("Average Resale Price ($)")
plt.title("HDB Resale Price Trend before ML Algorithms (2017 - 2025)")
plt.grid(True)

# Show the plot
plt.show()
