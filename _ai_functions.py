import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.metrics import r2_score


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series(
            [
                X[c].value_counts().index[0]
                if X[c].dtype == np.dtype("O")
                else X[c].median()
                for c in X
            ],
            index=X.columns,
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def xg(params, X_train, y_train):
    """
    Takes a list of model parameters, training X values, and training y values as input and returnes a
    trained XGBoost model.
    """
    if len(params) != 4 or params == None:
        params = [9, 1000, 0.035, True]
        print("Using default xg parameters")
    model = xgb.XGBRegressor(
        max_depth=params[0],
        n_estimators=params[1],
        learning_rate=params[2],
        tree_method="gpu_hist",
        single_precision_histogram=params[3],
    ).fit(X_train, y_train)
    return model


def rf(params, X_train, y_train):
    """
    Takes a list of model parameters, training X values, and training y values as input and returnes a
    trained random forest model.
    """
    
    if len(params) != 7 or params == None:
        params = [60, None, 12, 3, 1, True, True]
        print("Using default rf parameters")
    model = RandomForestRegressor(
        n_estimators=params[0],
        max_features=params[1],
        max_depth=params[2],
        min_samples_split=params[3],
        min_samples_leaf=params[4],
        bootstrap=params[5],
        oob_score=params[6],
    ).fit(X_train, y_train)
    return model


def run_model(method, df, feature_columns, target, test=[], params=None):
    """
    Function takes m*n pandas dataframe (df) and method (with optional parameters) as inputs and gives
    predictions as output.
    feature columns (lst) = list with length m with names of columns containing predictors
    target (str) = name of target variable

    Returns 1xn pandas dataframe of predicted values
    """
    big_X = df[feature_columns]
    # big_X_imputed = DataFrameImputer().fit_transform(big_X)
    big_Y = df[target]

    le = LabelEncoder()
    big_X_imputed = big_X[feature_columns].apply(LabelEncoder().fit_transform)

    if isinstance(test, pd.DataFrame) != True:
        X_train, X_test, y_train, y_test = train_test_split(
            big_X_imputed, big_Y, test_size=0.33, random_state=0
        )
    else:
        X_train = big_X_imputed
        X_test = test[feature_columns].apply(LabelEncoder().fit_transform)
        y_train = big_Y
        y_test = test["SWE"]

    # ___________________________
    start = time.time()
    if method == "xg":
        model = xg(params, X_train, y_train)

    elif method == "rf":
        model = rf(params, X_train, y_train)

    time2compute = time.time() - start
    # ______________________________

    predictions = model.predict(X_test)

    """print(
        "Calculation finished in "
        + str(time2compute)
        + " using "
        + method
        + " with an R2 score of "
        + str(r2_score(predictions, y_test))
    )"""


    return predictions, max(r2_score(predictions, y_test), 0)