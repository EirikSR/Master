import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import time
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# Function for the jonas logistic regression
def jonas_func(x, a, b, off):
    return a * x + b


def jonas(classes, Sdf, val_df):
    # Initiating elements before looping over results
    elev_lst = [[0, 1400], [1400, 2000], [2000, 100000]]
    month = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    params = []
    res = pd.DataFrame()

    # Timing the regression function

    start = time.time()

    # Looping through elevation classes
    for elev in elev_lst:

        # Looping through months of the year
        for i in range(13):
            df = Sdf[pd.to_datetime(Sdf["Date"]).dt.month == i].loc[
                (Sdf.Elev <= elev[-1]) & (Sdf.Elev > elev[0])
            ]
            val = val_df[pd.to_datetime(val_df["Date"]).dt.month == i].loc[
                (val_df.Elev <= elev[-1]) & (val_df.Elev > elev[0])
            ]

            # Excluding elevation/month combinations with too few measurements
            if df.shape[0] > 5:
                # Excluding high densities and dropping rows containing NaN values
                df = df.drop(df[df.density > 0.7].index).dropna()

                # Initiating arrays containging features for logistic regression
                x1_ar = np.array(df[["SD"]]).reshape(-1)
                y_ar = np.array(df[["density"]]).reshape(-1)

                # Initial parameters
                p0 = [100, 50, 10]

                # Fitting parameters
                fittedParameters, pcov = curve_fit(
                    jonas_func, x1_ar, y_ar, p0, maxfev=50000
                )

                # Applying model to validation data and converting results to SWE in mm
                val_ar = np.array(val[["SD"]]).reshape(-1)
                val[["Predicted"]] = jonas_func(val_ar, *fittedParameters) * val_ar * 10

                # Storing validation results
                if res.shape[0] == 0:
                    res = val
                else:
                    res = res.append(val)

                # Storing parameters as they are elevation/time specific
                params.append([fittedParameters, month[i - 1], elev])

    print("DURATION: ", time.time() - start)

    print(r2_score(res.Predicted, res.SWE))

    # Computing class specific offset
    offset = []
    res2 = pd.DataFrame()

    for i in range(len(classes)):
        # Dividing dataset into classes and computing
        rdf = res.loc[(res.snowclass == classes[i])]
        # Storing offsets in a list as they could be of interest later
        offset.append(np.mean(rdf.Predicted.values - rdf.SWE.values))
        # Adding offsett to val-results
        rdf["Predicted"] = rdf["Predicted"] + offset[i]

        # Storing results with offsett added
        if i == 0:
            res2 = rdf
        else:
            res2 = res2.append(rdf)

    return res2


# Function for the sturm logistic regression
def sturm_func(x, p0, k1, k2, pmax):
    return ((pmax - p0) * (1 - np.exp(k1 * x[0] - k2 * x[1]))) + p0


def sturm(classes, Sdf, val_df):
    results = pd.DataFrame()
    start = time.time()

    for x in classes:

        df = Sdf.loc[(Sdf.snowclass == x)]
        df = df.drop(df[df.density > 0.7].index)

        if df.shape[0] > 5:
            # Initiating parameters
            pmax = 0.5940
            p0 = 0.2
            k1 = 0.001
            k2 = 0.001
            p0 = [p0, k1, k2, pmax]

            # Creating arrays of logistic regression features for fitting
            x1_ar = np.array(df[["SD"]]).reshape(-1)
            x2_ar = np.array(df[["DOY"]]).reshape(-1) - 122
            y_ar = np.array(df[["density"]]).reshape(-1)

            # Fitting logistic regression
            fittedParameters, pcov = curve_fit(
                sturm_func, (x1_ar, x2_ar), y_ar, p0, maxfev=50000
            )

            # Creating arrays for validation
            val = val_df.loc[(val_df.snowclass == x)]
            val_ar = np.array(val[["SD"]]).reshape(-1)
            val_ar2 = np.array(val[["DOY"]]).reshape(-1) - 122

            # Computing modelled output and converting from density to SWE in mm
            val[["Predicted"]] = (
                sturm_func((val_ar, val_ar2), *fittedParameters) * val_ar * 10
            )

            # Storing results
            if results.shape[0] == 0:
                results = val
            else:
                results = results.append(val)
    print("DURATION: ", time.time() - start)

    print(r2_score(results.Predicted, results.SWE))

    return results


def logistic_regression(method, Sdf, val_df):

    # Computing density in g/cm^3
    Sdf["density"] = (Sdf[["SWE"]] / 10.0).div(Sdf.SD, axis=0)

    # Sturm snow classes
    classes = [
        "Tundra",
        "Boreal Forest",
        "Maritime",
        "Ephemoral",
        "Praire",
        "Mountain Forest",
    ]
    # Example, this could be an issue for some ERA5-derived datasets,
    # Ocean, Ice, and Fill are not used here
    Sdf.loc[(Sdf.snowclass == "Ocean"), "snowclass"] = "Maritime"

    if method.lower() == "sturm".lower():
        ret = sturm(classes, Sdf, val_df)

    elif method.lower() == "jonas".lower():
        ret = jonas(classes, Sdf, val_df)
    return ret


# Initiating datasets
Sdf = pd.read_csv("Ak_ttv_val.csv", header=0, index_col=0)
val_df = pd.read_csv("Ak_ttv_test.csv", header=0, index_col=0)

a = logistic_regression("stUrm", Sdf, val_df)
b = logistic_regression("Jonas", Sdf, val_df)

print(a, b)