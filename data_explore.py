import pandas as pd

# load the CSV from the data folder
df = pd.read_csv('data/car_details.csv')

# quick peek at data
print(df.head())
print(df.info())
print(df.describe())
