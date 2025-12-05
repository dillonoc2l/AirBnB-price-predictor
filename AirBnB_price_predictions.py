import pandas as pd
import numpy as np 
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import  r2_score, mean_absolute_error
from scipy.stats import randint, uniform
import joblib
import os

df = pd.read_csv("clean_listings.csv")


#need to convert categorical data in 'neighbourhood' and 'room_type' to numerical data
#do this using one-hot encoding

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output= False).set_output(transform='pandas') #one-hot encoder as pandas dataframe

cat_cols = ["neighbourhood", "room_type"]

ohe_df = ohe.fit_transform(df[cat_cols])

df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)




#function to test, train and score model
def score_model(df_check): #can be run after changing data
    


    X = df_check.drop(['price', 'id', 'host_id'],  axis=1)#create features
    y = np.log1p(df_check["price"])#log price to compress outliers (expensive properties)

    #split data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=32
    )

    model =  GradientBoostingRegressor()#select model and fit to data
    model = model.fit(X_train,y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log) #expm1 data to reverse log of price and predict actual price

    # Evaluate models predictions
    r2 = r2_score(np.expm1(y_test), y_pred)#expm1 - turn log prices into actual prices
    print("R2 Score:", r2)

    #cross validation
    cvScore = cross_val_score(model, X, y.values.ravel(), cv = 10)

    print('average CV Score:', np.mean(cvScore))

    return model, X_train, X_test, y_train, y_test, y_pred


print('Original:')#evaluate original model using function above
model, X_train, X_test, y_train, y_test, y_pred = score_model(df)
    



#Add distance to centre of london instead of Latitude and longitude

def haversine(lat1, lon1, lat2, lon2): #Formula to calculate distance between two points on a sphere (earth)
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

center_lat, center_lon = 51.509865, -0.118092 #London Centre rough coordinates



df['dist_to_center'] = haversine( # calculate airbnbs distance to central london
    df['latitude'], df['longitude'],
    center_lat, center_lon
)

df.drop(columns="latitude", axis = 1, inplace=True)#remove latitude and longitude as no longer needed
df.drop(columns="longitude", axis = 1, inplace=True)


print('Added distance to centre:')# evaluate model after data change to see if it imporved
model, X_train, X_test, y_train, y_test, y_pred = score_model(df) 





param_distributions = { # hyperparameter ranges for random search cv
    'n_estimators': randint(100, 500),          
    'max_depth': randint(3, 7),                 
    'learning_rate': uniform(0.05, 0.25),       
    'subsample': uniform(0.7, 0.3),            
    'validation_fraction': uniform(0.1, 0.2),  
    'min_samples_split': randint(2, 6),         
    'min_samples_leaf': randint(1, 4)           
}

random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(),
    param_distributions=param_distributions,
    n_iter=50,              # number of random combinations to try
    cv=3,                   # 3-fold cross-validation
    scoring='r2',           # evaluation metric
    n_jobs=-1,              # use all CPU cores
    random_state=42        # reproducibility
)

random_search.fit(X_train, y_train)

print("Best Score:", random_search.best_score_)
print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_




#
def evaluate_model(model, X_test, y_test):
    #Return R2 and MAE for a given model on test data
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    r2 = r2_score(np.expm1(y_test), y_pred)
    mae = mean_absolute_error(np.expm1(y_test), y_pred)
    return r2, mae

def save_best_model(current_model, X, y, model_path='best_airbnb_model.pkl'):
    """
    Compare the current model with the saved one using average CV score
    Saves the better model
    """
    # Evaluate current model
    print("\nEvaluating current model using cross-validation...")
    current_cv = np.mean(cross_val_score(current_model, X, y, cv=5))
    print(f"Current model average CV score: {current_cv:.4f}")

    # If no saved model exists, save current immediately
    if not os.path.exists(model_path):
        joblib.dump(current_model, model_path)
        print("No saved model found — current model saved as best model.")
        return

    # Load and evaluate saved model
    print("\nEvaluating saved model using cross-validation...")
    saved_model = joblib.load(model_path)
    saved_cv = np.mean(cross_val_score(saved_model, X, y, cv=5))
    print(f"Saved model average CV score: {saved_cv:.4f}")

    # Compare
    if current_cv > saved_cv:
        joblib.dump(current_model, model_path)
        print("\nCurrent model is BETTER — saved as new best model.")
    else:
        print("\nSaved model is still best — no change made.")


X = df.drop(['price', 'id', 'host_id'],  axis=1)
y = np.log1p(df["price"])

evaluate_model(best_model, X_test, y_test)
save_best_model(best_model, X, y)

