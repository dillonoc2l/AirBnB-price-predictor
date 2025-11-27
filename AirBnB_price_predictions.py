import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("clean_listings.csv")
#print(df.head())




#need to convert categorical data in 'neighbourhood' and 'room_type' to numerical data
#do this using one-hot encoding

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output= False).set_output(transform='pandas') #one-hot encoder as pandas dataframe

oheNeighbourhood = ohe.fit_transform(df[['neighbourhood']])#creating one-hot encoded dataframes for both features
oheRoomType = ohe.fit_transform(df[['room_type']])

df = pd.concat([df, oheNeighbourhood], axis = 1).drop(columns = ['neighbourhood'])  #concatinating dataframes to original dataframe 
df = pd.concat([df, oheRoomType], axis = 1).drop(columns = ['room_type'])           #and dropping unnecessary catagorical data




X = df.drop(['price', 'id', 'host_id'],  axis=1)#create features
y = df["price"]#create target  

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,y)

cvscore = cross_val_score(clf, X, y.values.ravel(), cv = 10)

print(cvscore)
