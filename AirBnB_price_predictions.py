import pandas as pd
import numpy as np 
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import pairwise_distances, r2_score, mean_absolute_error


df = pd.read_csv("clean_listings.csv")


#need to convert categorical data in 'neighbourhood' and 'room_type' to numerical data
#do this using one-hot encoding

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output= False).set_output(transform='pandas') #one-hot encoder as pandas dataframe

cat_cols = ["neighbourhood", "room_type"]

ohe_df = ohe.fit_transform(df[cat_cols])

df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)


def score_model(df_check):
    X = df_check.drop(['price', 'id', 'host_id'],  axis=1)#create features
    y = np.log1p(df_check["price"])#log price to compress outliers (expensive properties)

    #split data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=32
    )

    model =  GradientBoostingRegressor()
    model = model.fit(X_train,y_train)

    #y_pred = model.predict(X_test)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log) #expm1 data to reverse log of price and predict actual price

    # Evaluate
    r2 = r2_score(np.expm1(y_test), y_pred)#expm1 - turn log prices into actual prices
    print("R2 Score:", r2)

    #cross validation
    cvScore = cross_val_score(model, X, y.values.ravel(), cv = 5)

    print('average CV Score:', np.mean(cvScore))


print('Original:')
score_model(df)
    



#Add distance to centre of london instead of Latitude and longitude

def haversine(lat1, lon1, lat2, lon2): #Formula to calculate distance between two points on a sphere (earth)
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

center_lat, center_lon = 51.509865, -0.118092 #London Centre rough coordinates

df_added_centre = df.copy() #copy dataframe to make changes to

df_added_centre['dist_to_center'] = haversine(
    df_added_centre['latitude'], df_added_centre['longitude'],
    center_lat, center_lon
)

df_added_centre.drop(columns="latitude", axis = 1, inplace=True)#remove latitude and longitude as no longer needed
df_added_centre.drop(columns="longitude", axis = 1, inplace=True)

print('Added distance to centre:')
score_model(df_added_centre) # check to see if model improved

df = df_added_centre # add changes to original data frame




''' #gridsearch for best hyperparameters
param_grid = [{ 
    'max_depth' : [3,4,5,6], 
    'learning_rate' : [0.1, 0.2]
    #'validation_fraction' : [0.1,0.3,0.5]
    }]

grid_search = GridSearchCV(model, param_grid, cv=3, scoring = 'r2', n_jobs=-1 )

grid_search.fit(X_train,y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.cv_results_)

'''