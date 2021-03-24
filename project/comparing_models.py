import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def _model_train_forest(df,tag,test=False):
    """
    example funtion to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    """


    ## start timer for runtime
    time_start = time.time()

    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a random forest model
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_mse =  round(mean_squared_error(y_test,y_pred))
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    print("country", tag, "--  random forest eval  --")
    print("sme:", eval_mse)
    print("rsme:", eval_rmse)


def model_train_forest(data_dir,test=False):
    """
    funtion to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")

    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():

        if test and country not in ['all','united_kingdom']:
            continue

        _model_train_forest(df,country,test=test)
def _model_train_gradient_boost(df,tag,test=False):
    """
    example funtion to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    """


    ## start timer for runtime
    time_start = time.time()

    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a gradient boost regressor model
    param_grid_gb = {
    'gb__criterion': ['mse','mae'],
    'gb__n_estimators': [10,15,20,25]
    }

    pipe_gb = Pipeline(steps=[('scaler', StandardScaler()),
                              ('gb', GradientBoostingRegressor())])

    grid = GridSearchCV(pipe_gb, param_grid=param_grid_gb, cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_mse =  round(mean_squared_error(y_test,y_pred))
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    print("country", tag, "--  gradient boost regressor eval  --")
    print("sme:", eval_mse)
    print("rsme:", eval_rmse)


def model_train_gradient_boost(data_dir,test=False):
    """
    funtion to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")

    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():

        if test and country not in ['all','united_kingdom']:
            continue

        _model_train_gradient_boost(df,country,test=test)

if __name__ == "__main__":

    """
    basic test procedure for model_performance.py
    """
    data_dir = os.path.join(".","data","cs-train")

    ## train the model
    print("TRAINING MODEL Random Forest")
    model_train_forest(data_dir,test=True)

    print("TRAINING MODEL Gradient Boost")
    model_train_gradient_boost(data_dir,test=True)
