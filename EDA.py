import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("clean_listings.csv")

sns.set(style="whitegrid")

'''
#Basic overview of data
print(df.head())
print(df.info)
print(df.describe)
print(df.isna().sum())


#Plot price distribution
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df['price']), bins=50, kde=True)
plt.title("Distribution of Airbnb Prices")
plt.xlabel("Price (£)")
plt.show()
print(df['price'].skew())

'''

'''Price distribution was approximately symmetrical with a mild positive skew, indicating that most listings
fall in a typical range with a small number of higher-priced properties. This justified the use of a log
transform to stabilise variance and improve model performance.'''


#Plot relationships between features and price


dfSample = df.sample(400)


plt.figure(figsize=(8,5))
sns.scatterplot(data=dfSample, x='number_of_reviews', y='price')
plt.title("Reviews vs Price")
plt.show()

'''The above scatterplot shows there is no correlation between number_of_reviews adn price as 
Airbnbs with a high number of reviews can still be priced on the lower end of the scale
and Airbnbs with fewer reviews can still be priced highly'''


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='room_type', y='price')
plt.title("Room Type vs Price")
plt.show()

'''The above boxplot shows that private rooms are generally priced higher, entire home/apts can also be priced highly bu have fewer 
listings in the upper quartile. Hotel rooms are priced in the lower end and shared rooms can be priced even lower.'''


plt.figure(figsize=(12,6))
top10 = df.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top10.index, y=top10.values)
plt.xticks(rotation=45)
plt.title("Top 10 Most Expensive Neighbourhoods")
plt.ylabel("Average Price (£)")
plt.show()

'''The barchart above shows us that London areas that are close to central london have higher average prices. It also shows Lambeth has the highest average price
whis is likely due to it not only being central but also scenic, well-connected and tourist-heavy,'''