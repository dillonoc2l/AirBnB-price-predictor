#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("listings.csv")
#print(df.head)

#df.info()

#cleaning data

#drop columns not needed for data analysis and null columns
df.drop(columns="neighbourhood_group", inplace = True)
df.drop(columns="license", inplace = True)
df.drop(columns="last_review", inplace = True)
df.drop(columns="host_name", inplace = True)
df.drop(columns="name", inplace = True)

print(df.isna().sum())#print the number of null values in each column

raw_df = df#copy dataframe before removing columns

df = df.dropna(subset=['price'])#drop rows with null values in price column
df = df.dropna(subset=['reviews_per_month'])#drop rows with null values in reviews_per_month column


print(df.isna().sum())

print("\n")
print("duplicate rows: " + str(df.duplicated('id').sum()))#print the number of duplicate rows


df.to_csv('clean_listings.csv', index= False)

print(df.head())
