from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import psutil
import joblib

# Track memory usage before and after model training
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # in MB

# Track the time taken for training and testing the model
start_time = time.time()

# ----------- Loading & Prep DataFrame ---------------
# Load the cleaned dataset
df = pd.read_csv("cleanedHDB.csv")

#Here til ---- is prepping/preprocessing for the RandomForest
# Convert 'month' to datetime format for proper plotting
df['month'] = pd.to_datetime(df['month'])

df[["storey_min", "storey_max"]] = df["storey_range"].str.split(" TO ", expand=True).astype(int)
# Compute average storey
df["storey_avg"] = (df["storey_min"] + df["storey_max"]) / 2

# Categorize storey range
def categorize_storey(storey_avg):
    if storey_avg <= 4:
        return "Low"
    elif storey_avg <= 8:
        return "Middle"
    else:
        return "High"

df["storey_category"] = df["storey_avg"].apply(categorize_storey)

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
one_hot_encoded = encoder.fit_transform(df[['town', 'estate_type', 'storey_category']])
one_hot_columns = encoder.get_feature_names_out(['town', 'estate_type', 'storey_category'])
df_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)

# Concatenate the one-hot encoded columns with the original DataFrame
df = pd.concat([df, df_encoded], axis=1).drop(columns=['town', 'estate_type', 'storey_category'])

if 'resale_price' in df.columns:
    print("Column exists")
else:
    print("Column does not exist")

# ----------- Data Preprocessing ---------------
# Features include years + floor area + storey category + estate type + town + no. mrt stations
features = ["remaining_years", "floor_area_sqm", "mrt_station"] + list(one_hot_columns)
target = "resale_price" #target obvious lah eh

# Ensure the dataset contains only the selected columns idt this is necessary but good habit
df = df[features + [target] + ['month']]

upper_limit = df['resale_price'].quantile(0.99)
df = df[df['resale_price'] <= upper_limit]

# Prepare feature matrix (X) and target vector (y)
X = df.drop(columns=[target, 'month'])
y = df[target]

# Select only the features for future predictions
new_data = df[features]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#StandardScaler to scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------- Algorithm Implementation ---------------
# Initialize RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

# Train the model with the scaled training data
reg.fit(X_train_scaled, y_train)
y_predict = reg.predict(X_test_scaled)

# Scale the new data for future predictions using the trained scaler
new_data_scaled = scaler.transform(new_data)
future_predictions = reg.predict(new_data_scaled)

df['predicted_resale_price'] = future_predictions
df['year'] = df['month'].dt.year
df['future_year'] = df['year'] + 8  # Set future year for plotting purposes
# Calculate the average predicted resale price per future year
avg_predicted_prices = df.groupby('future_year')['predicted_resale_price'].mean()
avg_predicted_prices.index = avg_predicted_prices.index.astype(int)

# ----------- Performance Metrics ---------------
r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
print(avg_predicted_prices)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# ----------- Memory Usage ---------------
# Save the trained model to a file (using joblib)
model_filename = 'Random_Forest_model.pkl'
joblib.dump(reg, model_filename)
model_size = os.path.getsize(model_filename)
print(f"Model size: {model_size / 1024:.2f} KB")  # Size in KB

# Track the time taken for testing (prediction phase)
testing_start_time = time.time()
y_predict = reg.predict(X_test_scaled)
testing_end_time = time.time()

# Time taken for testing
testing_time = testing_end_time - testing_start_time

# Memory usage during training
memory_used = get_memory_usage()

# Track the time taken for training (model training phase)
training_end_time = time.time()
training_time = training_end_time - start_time

print(f"Time taken to train the model: {training_time:.2f} seconds")
print(f"Time taken to test the model: {testing_time:.2f} seconds")
print(f"Memory used during model training: {memory_used:.2f} MB")

# ----------- Visualisation ---------------
plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values,  linestyle='-', color='green',
         label='Avg Predicted Resale Price')
plt.xticks(avg_predicted_prices.index)
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by Decision Tree")
plt.legend()
plt.grid(True)
plt.show()
