import time
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def read_and_train():

    # read data
    df_origin = pd.read_csv("https://raw.githubusercontent.com/KuserOguzHan/mlops_1/main/hepsiburada.csv.csv")
    df_origin.head()

    df = df_origin.drop(["manufacturer"], axis=1)

    df.head()
    df.info()
    # Feature matrix
    X = df.iloc[:, 0:-1].values
    print(X.shape)
    print(X[:3])

    # Output variable
    y = df.iloc[:, -1]
    print(y.shape)
    print(y[:6])

    # split test train
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train model
    from sklearn.ensemble import RandomForestRegressor

    estimator = RandomForestRegressor(n_estimators=200)
    estimator.fit(X_train, y_train)

    # Test model
    y_pred = estimator.predict(X_test)
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print("R2: ".format(r2))

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(r2)

    # Save Model
    import joblib
    joblib.dump(estimator, "saved_models/randomforest_with_hepsiburada.pkl")

    # make predictions
    # Read models
    estimator_loaded = joblib.load("saved_models/randomforest_with_hepsiburada.pkl")

    # Prediction set
    X_manual_test = [[64.0, 4.0, 6.50, 3500, 8.0, 48.0, 2.0, 2.0, 2.0]]
    print("X_manual_test", X_manual_test)

    prediction = estimator_loaded.predict(X_manual_test)
    print("prediction", prediction)

