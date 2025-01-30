from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## I AINT DONE - Elfian ##

# Load the cleaned dataset
df = pd.read_csv("cleanedHDB.csv")
# Convert 'month' to datetime format for proper plotting
df['month'] = pd.to_datetime(df['month'])

# Encode categorical 'street_name' into numbers
label_encoder = LabelEncoder()
df['street_name'] = label_encoder.fit_transform(df['street_name'])

if 'resale_price' in df.columns:
    print("Column exists")
else:
    print("Column does not exist")

# according to my research(totally chatgpt), this 2 affects the resale value the most
features = ["remaining_years", "street_name"]
target = "resale_price"

# Ensure selected columns exist in the dataset
df = df[features + [target]]

# Split the data into training and testing sets
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_predict = model.predict(X_test)

# Evaluate using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_predict)
print(f"Mean Absolute Error: {mae}")

# Convert predictions into a DataFrame for visualization
df_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df_predicted = df_predicted.sort_index()  # Align with original index


plt.figure(figsize=(12, 6))
plt.plot(df_predicted.index, df_predicted['Actual'], label="Actual Prices", marker='o', linestyle='dashed')
plt.plot(df_predicted.index, df_predicted['Predicted'], label="Predicted Prices", marker='o', linestyle='solid', alpha=0.7)
plt.title("📈 Actual vs Predicted Resale Prices")
plt.xlabel("Index")
plt.ylabel("Resale Price (SGD)")
plt.legend()
plt.grid(True)
plt.show()