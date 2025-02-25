import pandas as pd

#Load the dataset
df = pd.read_csv('HDBData.csv')
df = df.dropna() #this part to remove empty cells
df = df.drop_duplicates() #this part to remove duplicates

df = df[df['flat_type'] != 'MULTI-GENERATION']

print("It worked lmao")
df.to_csv('cleanedHDB.csv', index=False)
