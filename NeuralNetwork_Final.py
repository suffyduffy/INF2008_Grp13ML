import time
import os
import psutil
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Function to track memory usage in MB (RAM usage)
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Convert bytes to MB

# Start tracking time and memory before training and testing the model
start_time = time.time()
start_memory = get_memory_usage()

#-------------Loading and Prep DataFrame-----------------
# Load the cleaned dataset
df = pd.read_csv("cleanedHDB.csv")

# Convert 'month' to datetime format
df['month'] = pd.to_datetime(df['month'])

# Extract storey min and max from 'storey_range' column
df[['storey_min', 'storey_max']] = df['storey_range'].str.split(' TO ', expand=True).astype(int)
df['storey_avg'] = (df['storey_min'] + df['storey_max']) / 2

# Function to categorize storey level
def categorize_storey(storey_avg):
    if storey_avg <= 4:
        return "Low"
    elif storey_avg <= 8:
        return "Middle"
    else:
        return "High"

df['storey_category'] = df['storey_avg'].apply(categorize_storey)

# Remove outliers from resale price using Interquartile Range (IQR)
Q1 = df['resale_price'].quantile(0.25)
Q3 = df['resale_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['resale_price'] >= lower_bound) & (df['resale_price'] <= upper_bound)]

#-----------------Feature Engineering------------------------------
# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
one_hot_encoded = encoder.fit_transform(df[['flat_type', 'estate_type', 'storey_category', 'flat_model', 'town']])
one_hot_columns = encoder.get_feature_names_out(['flat_type', 'estate_type', 'storey_category', 'flat_model', 'town'])
df_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)


# Merge encoded features into original dataframe and drop old categorical columns
df = pd.concat([df, df_encoded], axis=1).drop(columns=['flat_type', 'estate_type', 'storey_category', 'flat_model', 'town'])

# Define features (X) and target (y)
features = ["remaining_years", "floor_area_sqm", "mrt_station", "year"] + list(one_hot_columns)
target = "resale_price"

df = df[features + [target] + ['month']]

# Remove any NaN values
df = df.dropna()

# Reset index before applying ShuffleSplit
df = df.reset_index(drop=True)

# Apply ShuffleSplit to randomly sample 80% of data
split = ShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
for train_index, test_index in split.split(df):
    df_sampled = df.iloc[test_index]  # Select the test split from ShuffleSplit

# Prepare feature matrix (X) and target variable (y)
X = df_sampled.drop(columns=[target, 'month'])
y = df_sampled[target]

#------------- Splitting Data into Training and Testing Sets -----------------
# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#------------- Data Scaling ----------------------------
# Standardizing numerical features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit & transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

#------------- Define and Train Neural Network Model -----------------
# Initialize Multi-Layer Perceptron Regressor (MLPRegressor)
model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                     alpha=0.0005, batch_size=64, max_iter=75, random_state=42, verbose=True,
                     early_stopping=True, n_iter_no_change=5)

# Train the neural network model
model.fit(X_train_scaled, y_train)

#------------- Model Size Estimation -----------------
# Calculate total number of parameters in the trained model
total_params = sum(np.prod(w.shape) for w in model.coefs_) + sum(np.prod(b.shape) for b in model.intercepts_)
model_size_kb = (total_params * 8) / 1024  # Convert bytes to KB
print(f"Estimated Model Size: {model_size_kb:.2f} KB")

# Track training time and memory usage after model training
training_time = time.time() - start_time  # Training duration
training_memory = get_memory_usage() - start_memory  # Memory used during training

#------------- Model Testing & Prediction -----------------
# Start tracking time for testing phase
testing_start_time = time.time()

# Generate predictions on test set
y_pred = model.predict(X_test_scaled)

# Replace NaN predictions (if any) with the mean prediction value
y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))

# Track time taken for testing
testing_time = time.time() - testing_start_time

#------------- Model Evaluation Metrics -----------------
# Compute standard evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
r2 = r2_score(y_test, y_pred)  # R² Score

# Compute Adjusted R² Score
n = X_test.shape[0]  # Number of test samples
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Print Performance Metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Adjusted R² Score: {adjusted_r2:.4f}")
print(f"Time taken to train the model: {training_time:.2f} seconds")
print(f"Time taken to test the model: {testing_time:.2f} seconds")
print(f"Memory used during training: {training_memory:.2f} MB")

#------------- Future Resale Price Predictions -----------------
# Predict resale prices for the next 8 years
df['future_year'] = df['month'].dt.year + 8  # Predict 8 years into the future
new_data = df[features]
new_data_scaled = scaler.transform(new_data)
future_predictions = model.predict(new_data_scaled)

df['predicted_resale_price'] = future_predictions
avg_predicted_prices = df.groupby('future_year')['predicted_resale_price'].mean()

#------------- Visualization of Future Predictions -----------------
plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values, linestyle="-", color="blue", label="Avg Predicted Resale Price")
plt.xticks(avg_predicted_prices.index)
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by MLPRegressor (Neural Network)")
plt.legend()
plt.grid(True)
plt.show()

