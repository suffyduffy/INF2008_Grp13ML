import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import LinearSVR 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys

# Function to remove outliers using IQR
def remove_outliers_iqr_numpy(data, column):
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_indices = np.where((data[column] >= lower_bound) & (data[column] <= upper_bound))
    data_filtered = data.iloc[filtered_indices]
    return data_filtered

# Load the HDB dataset
df = pd.read_csv('cleanedHDB.csv')

# Remove outliers from 'resale_price'
df = remove_outliers_iqr_numpy(df, 'resale_price')

# Classify storey range
df[["storey_min", "storey_max"]] = df["storey_range"].str.split(" TO ", expand=True).astype(int)
df["storey_avg"] = (df["storey_min"] + df["storey_max"]) / 2

def categorize_storey(storey_avg):
    if storey_avg <= 4:
        return "Low"
    elif storey_avg <= 8:
        return "Middle"
    else:
        return "High"
        
df["storey_category"] = df["storey_avg"].apply(categorize_storey)

# Select relevant columns, including categorical features
features = ["remaining_years", "floor_area_sqm", "year", "flat_type", "storey_category", "flat_model", "town", "mrt_station"]
if "storey_range" in features:
    features.remove("storey_range")
features.append("storey_category")
target = "resale_price"
df = df[[*features, target]]

# --- Random Sampling ---
df = df.reset_index(drop=True)

split = ShuffleSplit(n_splits=1, test_size=0.8, random_state=42)

# Applying the split to your data
for train_index, test_index in split.split(df):
    df_sampled = df.loc[test_index]

# --- Data Preparation ---
X = df_sampled[features].copy() 
y = df_sampled[target].copy() 

# Remove NA values before one-hot encoding and scaling
X = X.dropna()
y = y.loc[X.index]

# One-hot encoding for categorical features
X = pd.get_dummies(X, columns=['flat_type', 'storey_category', 'flat_model', 'town'], drop_first=True, dtype=np.uint8)

# Normalize numerical features
numerical_features = ['remaining_years', 'floor_area_sqm', 'year', "mrt_station"]
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---
start_time = time.time()  # Record training start time

# Baseline Model
baseline_model = LinearSVR(random_state=42)  # Use LinearSVR for baseline
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
baseline_mse = mean_squared_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)

# Hyperparameter tuning using RandomizedSearchCV with KFold
svm_model = LinearSVR(random_state=42)  # Use LinearSVR for tuning
param_distributions = {
    'C': np.logspace(-2, 2, 5),
    'epsilon': [0, 0.1, 0.2, 0.5, 1],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 
    'max_iter': [5000, 10000, 20000]
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(svm_model, param_distributions, n_iter=20, cv=kf, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
random_search.fit(X, y)

# Get the best model from random search
best_svm_model = random_search.best_estimator_

end_time = time.time()  # Record training end time
training_time = end_time - start_time

# Get model size
model_size = sys.getsizeof(best_svm_model)

# Make predictions on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Compare Tuning Improvement
mse_improvement = baseline_mse - mse
r2_improvement = r2 - baseline_r2

print(f"Hyperparameter Tuning MSE Improvement: {mse_improvement:.4f}")
print(f"Hyperparameter Tuning R-squared Improvement: {r2_improvement:.4f}")

# --- Future Predictions with Multiple Scenarios ---

# Record testing start time
start_time_test = time.time()

latest_year = df_sampled['year'].max()
future_years = np.arange(latest_year + 0, latest_year + 9)
num_scenarios = 5  # Number of scenarios per year
all_future_predictions = []

for year in future_years:
    for _ in range(num_scenarios):  # Iterate to create multiple scenarios
        # 1. Randomly sample an existing data point
        scenario_data = df_sampled.sample(1, random_state=np.random.randint(0, 1000)).copy()  # Use a different random state for each scenario
        # Reset index to avoid duplicate indices later
        scenario_data = scenario_data.reset_index(drop=True) 

        # 2. Update the 'year' and potentially other numerical features
        scenario_data['year'] = year
        scenario_data['remaining_years'] = scenario_data['remaining_years'] + np.random.randint(-5, 5)  # Adjust remaining_years
        scenario_data['floor_area_sqm'] = scenario_data['floor_area_sqm'] * (1 + np.random.uniform(-0.05, 0.05))  # Adjust floor area

        # 3. One-hot encode using the fitted encoder, ensuring consistent columns
        # Get categorical data for encoding
        categorical_data = scenario_data[['flat_type', 'storey_category', 'flat_model', 'town']] 
        # Perform one-hot encoding
        encoded_data = pd.get_dummies(categorical_data, 
                                     columns=['flat_type', 'storey_category', 'flat_model', 'town'], 
                                     drop_first=True, dtype=np.uint8)  
        # Reindex the encoded data to match the columns of X_train, filling missing columns with 0
        # Exclude numerical features from reindexing
        encoded_data = encoded_data.reindex(columns=[col for col in X_train.columns if col not in numerical_features], fill_value=0) 

        scenario_data = scenario_data.drop(columns=['flat_type', 'storey_category', 'flat_model', 'town'])  # Drop original categorical columns
        scenario_data = pd.concat([scenario_data, encoded_data], axis=1)  # Concatenate encoded data

        # 4. Normalize using the fitted scaler, ensuring consistent order of numerical features
        scenario_data[numerical_features] = scaler.transform(scenario_data[numerical_features].reindex(columns=numerical_features, fill_value=0))

        # Added these lines to drop 'cluster' and 'resale_price' before prediction
        scenario_data = scenario_data.drop(columns=['cluster', 'resale_price'], errors='ignore')
         
        # 5. Make predictions
        future_prediction = best_svm_model.predict(scenario_data)
        all_future_predictions.extend(future_prediction)

# Create DataFrame for future predictions
future_predictions_df = pd.DataFrame({
    'year': np.repeat(future_years, num_scenarios),
    'predicted_price': all_future_predictions
})

# Record testing end time
end_time_test = time.time()

# Calculate testing time
testing_time = end_time_test - start_time_test

# Memory Usage
memory_usage = df.memory_usage(deep=True).sum()  # Memory used by DataFrame
print(f"Model Size: {model_size / 1024:.2f} KB")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Testing Time: {testing_time:.2f} seconds")
print(f"Memory Usage: {memory_usage / (1024 * 1024):.2f} MB")

# --- Performance Metrics and Fit Analysis ---
r2_train = r2_score(y_train, best_svm_model.predict(X_train)) # using best_sgd_model
r2_test = r2_score(y_test, y_pred)

mae_train = mean_absolute_error(y_train, best_svm_model.predict(X_train)) # using best_sgd_model
mae_test = mean_absolute_error(y_test, y_pred)

mse_train = mean_squared_error(y_train, best_svm_model.predict(X_train)) # using best_sgd_model
mse_test = mean_squared_error(y_test, y_pred)

# --- Visualization ---
plt.figure(figsize=(12, 6))

# Subplot for Training Data
plt.subplot(1, 2, 1)
plt.scatter(y_train, best_svm_model.predict(X_train), color='blue', label='Train Data') # using best_sgd_model
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label="Perfect Fit")
plt.title('Training Data: Underfit, Overfit or Goodfit')
plt.xlabel('Actual Resale Price')
plt.ylabel('Predicted Resale Price')
plt.legend()
plt.grid(True)

# Subplot for Test Data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='green', label='Test Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Fit")
plt.title('Testing Data: Underfit, Overfit or Goodfit')
plt.xlabel('Actual Resale Price')
plt.ylabel('Predicted Resale Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print model status based on MAE and R²
if mae_train < mae_test and r2_train > 0.9 and r2_test < 0.5:
    print("The model is overfitting.")
elif mae_train > mae_test and r2_train < 0.5 and r2_test < 0.5:
    print("The model is underfitting.")
else:
    print("The model is a good fit.")

# --- Visualization ---

# Calculate average predicted resale price for each future year
avg_predicted_prices = future_predictions_df.groupby('year')['predicted_price'].mean()

# Plot the average predicted resale price over the upcoming years
plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values, linestyle='-', color='green', label='Avg Predicted Resale Price')
plt.xticks(avg_predicted_prices.index)  # Set x-axis ticks to the future years
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by SVR Model")  
plt.legend()
plt.grid(True)
plt.show()

# Box plot: Future Price Distribution per Year
plt.figure(figsize=(10, 6))
plt.boxplot([future_predictions_df[future_predictions_df['year'] == year]['predicted_price']
             for year in future_years], tick_labels=future_years)
plt.xlabel("Year")
plt.ylabel("Predicted Resale Price")
plt.title("Future Resale Price Distribution per Year")
plt.grid(True)
plt.show()

# Heatmap (Correlation Matrix)
numerical_features_for_corr = ['remaining_years', 'floor_area_sqm', 'resale_price','year']
correlation_matrix = df_sampled[numerical_features_for_corr].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df_sampled['floor_area_sqm'], df_sampled['resale_price'])
plt.xlabel('Floor Area (sqm)')
plt.ylabel('Resale Price')
plt.title('Scatter Plot: Floor Area vs. Resale Price')
plt.grid(True)
plt.show()

# Scatter Plot with K-means Clustering
plt.figure(figsize=(8, 6))
n_clusters = 3  # Choose the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_sampled['cluster'] = kmeans.fit_predict(df_sampled[['floor_area_sqm', 'resale_price']])

# Create scatter plot with colored clusters
scatter = plt.scatter(df_sampled['floor_area_sqm'], df_sampled['resale_price'], c=df_sampled['cluster'], cmap='viridis')
plt.xlabel('Floor Area (sqm)')
plt.ylabel('Resale Price')
plt.title('Scatter Plot: Floor Area vs. Resale Price with K-means Clustering')
plt.grid(True)

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(df_sampled['resale_price'], bins=20)
plt.xlabel('Resale Price')
plt.ylabel('Frequency')
plt.title('Histogram: Resale Price Distribution')
plt.grid(True)
plt.show()

# Calculate average resale price for each flat type
avg_price_by_flat_type = df_sampled.groupby('flat_type')['resale_price'].mean()
plt.figure(figsize=(10, 6))
plt.bar(avg_price_by_flat_type.index, avg_price_by_flat_type.values)
plt.xlabel('Flat Type')
plt.ylabel('Average Resale Price')
plt.title('Bar Chart: Average Resale Price by Flat Type')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()

# Feature Importance
feature_importances = pd.Series(best_svm_model.coef_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')  # Top 10 features
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.show()

# Residual (Error) Analysis
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at 0
plt.show()

# Choose a feature to analyze (e.g., 'floor_area_sqm')
feature_values = X_test['floor_area_sqm']

plt.figure(figsize=(8, 6))
plt.scatter(feature_values, residuals)
plt.xlabel("Floor Area (sqm)")
plt.ylabel("Residuals")
plt.title("Residuals vs. Floor Area")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
