import pandas as pd

#Load the dataset
df = pd.read_csv('HDBData.csv')
df = df.dropna() #this part to remove empty cells
df = df.drop_duplicates() #this part to remove duplicates

#Filter out rows from 2017 to 2019 as data might be too old
df = df[(df['month'] < '2017-01') | (df['month'] >= '2020-01')]

#Display the first few rows of the dataset
print(df.head())

df.to_csv('cleanedHDB.csv', index=False)