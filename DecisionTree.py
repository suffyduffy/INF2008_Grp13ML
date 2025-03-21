import time
import os
import psutil
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Track memory usage before and after model training
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # in MB
# Track the time taken for training and testing the model
start_time = time.time()

# ----------- Loading & Prep DataFrame ---------------
# Load the cleaned dataset
df = pd.read_csv("cleanedHDB.csv")

# Convert 'month' to datetime format for proper plotting
df['month'] = pd.to_datetime(df['month'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the 'town' column
df['town_encoded'] = label_encoder.fit_transform(df['town'])

if 'resale_price' in df.columns:
    print("Column exists")
else:
    print("Column does not exist")

# ----------- Data Preprocessing ---------------
# Features that affect resale value based on research on goo0gle
features = ["remaining_years", "floor_area_sqm", "town_encoded"]
target = "resale_price"

# Ensure the dataset contains only the selected columns
df = df[features + [target] + ['month']]

# Remove outliers by filtering resale prices beyond the 99th percentile
upper_limit = df['resale_price'].quantile(0.99)
df = df[df['resale_price'] <= upper_limit]

# Prepare feature matrix (X) and target vector (y)
X = df.drop(columns=[target, 'month'])
y = df[target]

# Select only the features for future predictions
new_data = df[features]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler to scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------- Algorithm Implementation ---------------
# Initialize the DecisionTreeRegressor model
reg = DecisionTreeRegressor(max_depth=10, random_state=42)

# Train the model with the scaled training data
reg.fit(X_train_scaled, y_train)
y_predict = reg.predict(X_test_scaled)

# Time taken for training
training_time = time.time() - start_time  # in seconds

# Scale the new data for future predictions using the trained scaler
new_data_scaled = scaler.transform(new_data)
future_predictions = reg.predict(new_data_scaled)

df['predicted_resale_price'] = future_predictions
df['year'] = df['month'].dt.year
df['future_year'] = df['year'] + 8  # Set future year for plotting purposes
# Calculate the average predicted resale price per future year
avg_predicted_prices = df.groupby('future_year')['predicted_resale_price'].mean()
avg_predicted_prices.index = avg_predicted_prices.index.astype(int)

# Print the average predicted prices per year and performance metrics
print(avg_predicted_prices)

# ----------- Performance Metrics ---------------
r2 = r2_score(y_test, y_predict)
print(f"R² Score: {r2:.4f}")

# ----------- Memory Usage ---------------
# Save the trained model to a file (using joblib)
model_filename = 'decision_tree_model.pkl'
joblib.dump(reg, model_filename)

# Check the size of the saved model file
model_size = os.path.getsize(model_filename)

# Print the model size in bytes
print(f"Model size: {model_size / 1024:.2f} KB")  # Size in KB

# Track the time taken for testing (prediction phase)
testing_start_time = time.time()
y_predict = reg.predict(X_test_scaled)
testing_end_time = time.time()

# Time taken for testing
testing_time = testing_end_time - testing_start_time  # in seconds

# Memory usage during training
memory_used = get_memory_usage()

# Print time and memory usage
print(f"Time taken to train the model: {training_time:.2f} seconds")
print(f"Time taken to test the model: {testing_time:.2f} seconds")
print(f"Memory used during model training: {memory_used:.2f} MB")

# ----------- Visualisation ---------------
# Plot the average predicted resale price over the upcoming years
plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values, linestyle='-', color='green',
         label='Avg Predicted Resale Price')
plt.xticks(avg_predicted_prices.index)
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by Decision Tree")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Actual vs Predicted Resale Prices
plt.figure(figsize=(8, 6))  # Increase the figure size for better readability
plt.scatter(y_test, y_predict, alpha=0.6, s=30, c='blue')  # Add transparency, adjust point size

# Add a red dashed line representing perfect prediction (y = x)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

# Adding gridlines to make the graph more readable
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adding the R² score as text on the plot
r2_text = f'R² = {r2:.4f}'
plt.text(0.05, 0.95, r2_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Customize axis labels and title
plt.xlabel('Actual Resale Price (in millions)', fontsize=14)
plt.ylabel('Predicted Resale Price (in millions)', fontsize=14)
plt.title('Actual vs Predicted Resale Prices (Decision Tree Model)', fontsize=16)

# Show the plot
plt.show()


