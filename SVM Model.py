import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR 

# Load the HDB dataset
df = pd.read_csv('cleanedHDB_shortened.csv')

# Select relevant columns, including 'room_type'
features = ["remaining_years", "floor_area_sqm", "flat_type"]  
target = "resale_price"
df = df[[*features, target]]

# Split data into features (X) and target (y)
X = df[features].values  
y = df[target].values

# Normalize numerical features (remaining_years and floor_area_sqm)
scaler = StandardScaler()
X[:, :2] = scaler.fit_transform(X[:, :2])  # Apply scaling to numerical features only

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Before One-Hot Encoding, convert 'flat_type' to numerical using Label Encoding
label_encoder = LabelEncoder()
X[:, 2] = label_encoder.fit_transform(X[:, 2]) # Apply Label Encoding

# One-Hot Encoding for Room Type
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Create a OneHotEncoder object
encoded_flat_type_train = encoder.fit_transform(X_train[:, 2].reshape(-1, 1))  # Fit and transform on training data
encoded_flat_type_test = encoder.transform(X_test[:, 2].reshape(-1, 1))  # Transform testing data

# Replace the original 'room_type' column with the encoded columns
X_train = np.concatenate([X_train[:, :2], encoded_flat_type_train], axis=1)
X_test = np.concatenate([X_test[:, :2], encoded_flat_type_test], axis=1)

# Use scikit-learn's SVR
svm_model = SVR() 

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}  # Define parameter grid
grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_svm_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE
mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
r2 = r2_score(y_test, y_pred)  # Calculate R-squared

# Calculate Adjusted R-squared
n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"SVM Model Mean Squared Error (MSE): {mse:.4f}")
print(f"SVM Model Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"SVM Model Mean Absolute Error (MAE): {mae:.4f}")
print(f"SVM Model R-squared (R2): {r2:.4f}")
print(f"SVM Model Adjusted R-squared: {adjusted_r2:.4f}")