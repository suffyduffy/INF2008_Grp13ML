import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# Function to remove outliers using IQR
def remove_outliers_iqr_numpy(data, column):
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_filtered = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data_filtered

# Function to generate prediction intervals using bootstrapping
def get_prediction_intervals(model, X, percentile=95):
    n_bootstraps = 1000
    bootstrap_predictions = []
    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap = X[bootstrap_indices]
        bootstrap_pred = model.predict(X_bootstrap)
        bootstrap_predictions.append(bootstrap_pred)
    bootstrap_predictions = np.array(bootstrap_predictions)
    lower_bound = np.percentile(bootstrap_predictions, (100 - percentile) / 2, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, 100 - (100 - percentile) / 2, axis=0)
    return lower_bound, upper_bound

# Load the HDB dataset
df = pd.read_csv('cleanedHDB.csv')

# Remove outliers from 'resale_price'
df = remove_outliers_iqr_numpy(df, 'resale_price')

# Select relevant columns, including 'month' for year calculation
features = ["remaining_years", "floor_area_sqm", "month"]
target = "resale_price"
df = df[[*features, target]]

# --- Stratified Random Sampling ---
strata_variables = ['floor_area_sqm', 'remaining_years', 'month']
df['stratum'] = df[strata_variables].astype(str).agg('-'.join, axis=1)
df_sampled = df.groupby('stratum', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42), include_groups=False)

# Convert 'month' column to numerical representation (e.g., year)
df_sampled['year'] = pd.to_datetime(df_sampled['month']).dt.year 
features = ["remaining_years", "floor_area_sqm", "year"] 

# --- Data Preparation ---
X = df_sampled[features]
y = df_sampled[target]

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---
svm_model = SVR()

# Hyperparameter tuning using RandomizedSearchCV with KFold
param_distributions = {'C': np.logspace(-1, 1, 3), 'kernel': ['linear', 'rbf']}
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Define KFold with 5 splits
random_search = RandomizedSearchCV(svm_model, param_distributions, n_iter=6, cv=kf, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X, y)  # Fit on the entire dataset (X, y)

# Get the best model from random search
best_svm_model = random_search.best_estimator_

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

# --- Future Predictions with Multiple Scenarios ---
latest_year = df_sampled['year'].max()
future_years = np.arange(latest_year + 1, latest_year + 6)
num_scenarios = 5
all_future_predictions = []

for year in future_years:
    remaining_years_scenarios = np.random.randint(40, 60, size=num_scenarios)
    floor_area_sqm_scenarios = np.random.randint(90, 120, size=num_scenarios)
    future_data_year = pd.DataFrame({
        'remaining_years': remaining_years_scenarios,
        'floor_area_sqm': floor_area_sqm_scenarios,
        'year': [year] * num_scenarios
    })
    future_data_scaled = scaler.transform(future_data_year[features])
    future_predictions_year = best_svm_model.predict(future_data_scaled)
    all_future_predictions.extend(future_predictions_year)

future_predictions_df = pd.DataFrame({
    'year': np.repeat(future_years, num_scenarios),
    'predicted_price': all_future_predictions
})

# --- Visualization ---

# Box plot: Future Price Distribution per Year
plt.figure(figsize=(10, 6))
plt.boxplot([future_predictions_df[future_predictions_df['year'] == year]['predicted_price']
             for year in future_years], tick_labels=future_years)
plt.xlabel("Year")
plt.ylabel("Predicted Resale Price")
plt.title("Future Resale Price Distribution per Year")
plt.grid(True)
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='year', y='predicted_price', hue='year', data=future_predictions_df, palette="Set3", legend=False)
plt.xlabel("Year")
plt.ylabel("Predicted Resale Price")
plt.title("Distribution of Predicted Resale Prices per Year")
plt.grid(True)
plt.show()

# Scatter Plots
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], y_pred, alpha=0.5, color='blue', label='Predicted Price')
plt.scatter(X_test[:, 0], y_test, alpha=0.5, color='red', label='Actual Price')  
plt.xlabel("Remaining Years (Test Data)")
plt.ylabel("Resale Price")  
plt.title("Predicted vs. Actual Resale Price (Test Data)")
plt.legend()  
plt.grid(True)
plt.show()

# Line Plot with Confidence Intervals and Mean
plt.figure(figsize=(10, 6))

# Plot median predicted price
plt.plot(future_years, future_predictions_df.groupby('year')['predicted_price'].median(), 
         marker='o', linestyle='-', label="Median Predicted Price")

# Calculate and add confidence intervals using bootstrapping
lower_bound, upper_bound = get_prediction_intervals(best_svm_model, future_data_scaled, percentile=95)
plt.fill_between(future_years, lower_bound, upper_bound, alpha=0.2, label="Confidence Interval (95%)")

# Calculate and plot mean predicted price
plt.plot(future_years, future_predictions_df.groupby('year')['predicted_price'].mean(), 
         linestyle='--', label="Mean Predicted Price")
plt.xlabel("Year")
plt.ylabel("Predicted Resale Price")
plt.title("Predicted Resale Price Trend over Time")
plt.legend()
plt.grid(True)
plt.show()

# Histograms
plt.figure(figsize=(8, 6))
plt.hist(future_predictions_df['predicted_price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Predicted Resale Price")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Resale Prices")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_test - y_pred, bins=20, color='salmon', edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.grid(True)
plt.show()

# Heatmap (Correlation Matrix)
correlation_matrix = df_sampled[features + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
