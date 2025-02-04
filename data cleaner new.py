import pandas as pd

#Load the dataset
df = pd.read_csv('HDBData.csv')
df = df.dropna() #this part to remove empty cells
df = df.drop_duplicates() #this part to remove duplicates


#Filter out rows from 2017 to 2020 as data might be too old
df = df[(df['month'] < '2017-01') | (df['month'] >= '2021-01')]

#Filter out column in flat_type: MULTI-GENERATION, as the data is too small (39 rows)
df = df[df['flat_type'] != 'MULTI-GENERATION']

print("It worked lmao")
df.to_csv('cleanedHDB.csv', index=False)

#IF GOT ANY NEW DISCOVERY TO EDIT REDUCE THE DATASET EDIT HERE
