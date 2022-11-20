from ai_functions import run_model
import pandas as pd

feature_columns = [
    "SD",
    "Elev",
    "Lat",
    "Long",
    "DOY",
    "days without snow",
    "number frost-defrost",
    "accum pos degrees",
    "average age SC",
    "number layer",
    "accum solid precip",
    "accum solid precip in last 10 days",
    "total precip last 10 days",
    "average temp last 6 days",
    # "snowclass"
]

df = pd.read_csv("ALASKA_Super_X.csv").iloc[0:100]
test = pd.read_csv("ALASKA_Super_X.csv").iloc[2:300]
target = "SWE"

"""
XG Params = [max_depth,                 # int
             n_estimators,              # int
            learning_rate,              # [0, 1]
            single_precision_histogram] # Bool  
            #Needs to be list with length 4"

RF Params = [n_estimators,       # int
            max_features,        # [1, len(features)], None for all features
            max_depth,           # int
            min_samples_split,   # int
            min_samples_leaf,    # int
            bootstrap,           # Bool
            oob_score]           # Bool
            #Must be list with length 7
"""
method = "rf"
params = []  # [3, None, 3, 3, 1, True, True] #[] for default


# Example of model run using default parameters and supplied test data

result = run_model(method, df, feature_columns, target, params=params, test=test)

print(result)


# Example of parameter optimalization:

lst = []

for i in range(1, 10):
    for j in range(5, 51, 5):
        params = [j, None, i, 3, 1, True, True]
        lst.append(
            [
                i,
                j,
                run_model(method, df, feature_columns, target, params=params, test=[])[
                    -1
                ],
            ]
        )

print(lst)