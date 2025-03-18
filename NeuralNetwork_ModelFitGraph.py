import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("cleanedHDB.csv")

# Convert 'month' to datetime format
df['month'] = pd.to_datetime(df['month'])

# Extract storey min and max
df[['storey_min', 'storey_max']] = df['storey_range'].str.split(' TO ', expand=True).astype(int)
df['storey_avg'] = (df['storey_min'] + df['storey_max']) / 2

# Categorize storey range
def categorize_storey(storey_avg):
    if storey_avg <= 4:
        return "Low"
    elif storey_avg <= 8:
        return "Middle"
    else:
        return "High"

df['storey_category'] = df['storey_avg'].apply(categorize_storey)

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
one_hot_encoded = encoder.fit_transform(df[['flat_type', 'estate_type', 'storey_category', 'flat_model', 'town']])
one_hot_columns = encoder.get_feature_names_out(['flat_type', 'estate_type', 'storey_category', 'flat_model', 'town'])
df_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)

df = pd.concat([df, df_encoded], axis=1).drop(columns=['flat_type', 'estate_type', 'storey_category', 'flat_model', 'town'])

# Define features and target
features = ["remaining_years", "floor_area_sqm", "mrt_station", "year"] + list(one_hot_columns)
target = "resale_price"

df = df[features + [target] + ['month']]
df = df.dropna()  # Remove NaN values

# Prepare feature matrix (X) and target vector (y)
X = df.drop(columns=[target, 'month'])
y = df[target]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Neural Network (MLPRegressor)
model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                     alpha=0.0005, batch_size=64, max_iter=75, random_state=42, verbose=True,
                     early_stopping=True, n_iter_no_change=5)

# Train the model
model.fit(X_train_scaled, y_train)

# Performance Metrics for Training and Testing
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Visualizing Model Fit: Scatter Plots for Training & Test Data
plt.figure(figsize=(12, 6))

# Subplot for Training Data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5, label='Train Data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label="Perfect Fit")
plt.title('Training Data: Underfit, Overfit or Good Fit')
plt.xlabel('Actual Resale Price')
plt.ylabel('Predicted Resale Price')
plt.legend()
plt.grid(True)

# Subplot for Test Data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.5, label='Test Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Fit")
plt.title('Testing Data: Underfit, Overfit or Good Fit')
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
