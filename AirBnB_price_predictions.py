import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold

df = pd.read_csv("clean_listings.csv")
#print(df.head())

X = df.drop('price', axis=1)#create features
y = df["price"]


print(X.shape)
print(y.shape)

#create 10 folds for cross validation
#cross validation splits the data into k different folds. Use k-1 folds for training and the remaining folds fo testing
#process repeated k times using different fold each time

kf = KFold(n_splits=10, shuffle= True, random_state=22)#fixed random seed of 22
train_scores = []#accuracy scores of folds
test_scores = []