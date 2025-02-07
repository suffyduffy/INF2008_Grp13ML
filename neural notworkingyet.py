# I AM USING https://www.analyticsvidhya.com/blog/2021/10/implementing-artificial-neural-networkclassification-in-python-from-scratch/
# AS REFERENCE COZ CHATGPT SUK ASS - HAMIZI
# NOT COMPLETE NIG

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Loading Dataset
data = pd.read_csv("cleanedHDB.csv")

# Separate features (X) and target (y)
X = data.iloc[:, 0:-1].values
print("Before preprocessing:", X)  # Show the data before processing

# Label encoding for a specific column (assuming column 2 is categorical)
LE1 = LabelEncoder()
X[:, 2] = np.array(LE1.fit_transform(X[:, 2]))
print("After label encoding:", X)  # Show the data after label encoding

# Clean the 'flat_type' column
# Either require to change the data set or adjust it to the main data cleaner.py CUH
# Extract only the numeric part of 'flat_type' (e.g., '2 ROOM' -> '2')
data['flat_type'] = data['flat_type'].str.extract('(\d+)', expand=False)

# Verify unique values after cleaning
print("Unique flat_type values after cleaning:", data['flat_type'].unique())

# Handle any missing values in the 'flat_type' column (if there are any)
data['flat_type'] = data['flat_type'].fillna(0).astype(int)  # Referring to this one from line 25 ><

# Now apply the OneHotEncoder to the cleaned 'flat_type' column
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['flat_type'])],  # Apply OneHotEncoder to 'flat_type'
    remainder="passthrough"  # Keep other columns as they are
)

X = np.array(ct.fit_transform(data))
print("After one-hot encoding:", X)
