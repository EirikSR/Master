from cv2 import AKAZE_create
import pandas as pd
import scipy.stats
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.transforms as transforms

classes = [
    "Maritime",
    "Mountain Forest",
    "Ephemoral",
    "Praire",
    "Boreal Forest",
    "Tundra",
    # "Ice",
    # "Ocean",
    # "Fill",
]


def load_data():
    it_xg_N = pd.read_csv("xg_results_N_est.csv", index_col=0)
    it_xg_dp = pd.read_csv("xg_results_depth.csv", index_col=0)
    it_xg_lr = pd.read_csv("xg_results_lr.csv", index_col=0)

    it_rf_N = pd.read_csv("rf_results_N_est.csv", index_col=0)
    it_rf_dp = pd.read_csv("rf_results_depth.csv", index_col=0)
    it_rf_lf = pd.read_csv("rf_results_leaf.csv", index_col=0)

    xg_r = pd.read_csv("xg_results.csv", index_col=0)
    rf_r = pd.read_csv("rf_results.csv", index_col=0)
    jn_r = pd.read_csv("jonas_results.csv", index_col=0)
    st_r = pd.read_csv("sturm_results.csv", index_col=0)
    mlp_r = pd.read_csv("mlp_results2.csv", index_col=0)
    # mlp_r["Predicted"] = mlp_r["pred"]

    xg_rAK = pd.read_csv("xg_results_AK.csv", index_col=0)
    rf_rAK = pd.read_csv("rf_results_AK.csv", index_col=0)
    jn_rAK = pd.read_csv("jonas_results_AK.csv", index_col=0)
    st_rAK = pd.read_csv("sturm_results_AK.csv", index_col=0)
    mlp_rAK = pd.read_csv("mlp_results_AK.csv", index_col=0)

    xg_rAKs = pd.read_csv("xg_results_AK_sorted.csv", index_col=0)
    rf_rAKs = pd.read_csv("rf_results_AK_sorted.csv", index_col=0)
    jn_rAKs = pd.read_csv("jonas_results_AK_sorted.csv", index_col=0)
    st_rAKs = pd.read_csv("sturm_results_AK_sorted.csv", index_col=0)
    mlp_rAKs = pd.read_csv("mlp_results_AK_sorted.csv", index_col=0)

    # mlp_rAK["Predicted"] = mlp_rAK["pred"]

    xg_rs = pd.read_csv("xg_results_sorted.csv", index_col=0)
    rf_rs = pd.read_csv("rf_results_sorted.csv", index_col=0)
    mlp_rs = pd.read_csv("mlp_results_sorted.csv", index_col=0)
    jn_rs = pd.read_csv("jonas_results_sorted.csv", index_col=0)
    st_rs = pd.read_csv("sturm_results_sorted.csv", index_col=0)
    mlp_rs["Predicted"] = mlp_rs["pred"]

    xg_it_AK_r = pd.read_csv("xg_results_ALASKA_IT.csv")
    rf_it_AK_r = pd.read_csv("xg_results_ALASKA_IT.csv")

    train = pd.read_csv("SX_train_clean.csv", index_col=0)
    val = pd.read_csv("SX_val_clean.csv", index_col=0)
    val_AK = pd.read_csv("ALASKA_Super_X.csv", index_col=0)


def MBE(y_true, y_pred):
    """
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Bias score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = y_true - y_pred
    mbe = diff.mean()
    return mbe


# Calculates R2, (R)MSE, MAE, and MBE from predicted and measured values
def accuracy_score(predictions, y_test):
    mse = mean_squared_error(predictions, y_test)
    print("MSE: ", mse)

    rms = mean_squared_error(y_test, predictions, squared=False)
    print("RMSE: ", rms)

    mae = mean_absolute_error(y_test, predictions)
    print("MAE: ", mae)

    mbe = MBE(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, rms, mae, mbe, r2


# Bar plot of snow measurement snowclass distribution
def bar_classes(ax, classes, df):
    train_lst = []
    for c in classes:
        train_lst.append(len(df.loc[(df.snowclass == c)]))
    ax.bar(
        np.arange(len(train_lst)),
        train_lst,
        align="center",
        # density=True,
        width=0.8,
        facecolor="gray",
        # alpha=0.5,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.xticks(np.arange(len(train_lst)), classes, rotation=10)
    ax.ylabel("Number of record")
    ax.title("Dataset snowclass distribution")


# Delete
def rf_it_AK_plot():
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()

    lst = []
    ilst = [4, 6, 8, 10, 12, 14, 16, 18]
    # ilst = [4, 8, 12, 16, 20, 24, 30]
    for i in ilst:
        df = pd.read_csv("IT_RES/RF_none_results_N_est_D" + str(i) + ".csv")
        lst.append(df)

    for df, i in zip(lst, ilst):
        df.loc[df.r2 < 0] = 0

        axs[0, 0].plot(df.N, df.r2, label=str(i))

    axs[0, 0].set_xlabel("Number of Estimators", fontsize=14)
    axs[0, 0].set_ylabel("R2 Score", fontsize=14)
    axs[0, 0].grid()
    axs[0, 0].legend(title="Layers:", loc="lower right")

    for df, i in zip(lst, ilst):
        df.loc[df.rms < 0] = 0

        axs[0, 1].plot(df.N, df.rms, label=str(i))
    axs[0, 1].set_xlabel("Number of Estimators", fontsize=14)
    axs[0, 1].set_ylabel("RMSE (mm)", fontsize=14)
    axs[0, 1].legend(title="Layers:", loc="upper right")
    axs[0, 1].grid()

    for df, i in zip(lst, ilst):
        df.loc[df.mae < 0] = 0

        axs[1, 0].plot(df.N, df.mae, label=str(i))

    axs[1, 0].set_xlabel("Number of Estimators", fontsize=14)
    axs[1, 0].set_ylabel("MAE (mm)", fontsize=14)
    axs[1, 0].legend(title="Layers:", loc="upper right")
    axs[1, 0].grid()

    for df, i in zip(lst, ilst):
        # df.loc[df.mbe < 0] = 0

        axs[1, 1].plot(df.N, df.mbe, label=str(i))

    axs[1, 1].set_xlabel("Number of Estimators", fontsize=14)
    axs[1, 1].set_ylabel("MBE (mm)", fontsize=14)
    axs[1, 1].legend(title="Layers:", loc="lower right")
    axs[1, 1].grid()
    axs[0, 0].set_title("a")
    axs[0, 1].set_title("b")
    axs[1, 0].set_title("c")
    axs[1, 1].set_title("d")
    fig.suptitle("Parameter Tuning of RF Model Using Alaskan Validation", fontsize=14)
    plt.show()


# Plotting training itterations of rf/xg models
def xg_it_AK_plot(
    rf=False, Title="Parameter Tuning of XGB Model using USCN Validation"
):
    x_label = "Number of Estimators"
    y_label = ["R2 Score", "RMSE (mm)", "MAE (mm)", "MBE (mm)"]
    legend = "Layers:"
    title = ["a", "b", "c", "d"]

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    lst = []

    if rf:
        ilst = [4, 6, 8, 10, 12, 14, 16, 18]
        for i in ilst:
            df = pd.read_csv("IT_RES/RF_none_results_N_est_D" + str(i) + ".csv")
            lst.append(df)
        i = 0
    else:
        for i in range(2, 11, 1):
            df = pd.read_csv("IT_RES/USCN_xg_results_N_est_D" + str(i) + ".csv")
            lst.append(df)
        i = 0
    for df in lst:
        adf = df.loc[df.r2 < 0] = 0
        axs[0, 0].plot(adf.N, adf.r2, label=str(i))

        adf = df.loc[df.rms < 0] = 0
        axs[0, 1].plot(adf.N, adf.rms, label=str(i))

        adf = df.loc[df.mae < 0] = 0
        axs[1, 0].plot(adf.N, adf.mae, label=str(i))

        axs[1, 1].plot(df.N, df.mbe, label=str(i))
        i = i + 1

    j = 0
    for x in range(2):
        for y in range(2):
            axs[x, y].set_xlabel(x_label, fontsize=12)
            axs[x, y].set_xlabel(y_label[j], fontsize=12)
            axs[x, y].legend(legend)
            axs[x, y].set_title(title[j])
            axs[x, y].grid()

    fig.suptitle(
        Title,
        fontsize=14,
    )
    plt.show()


def it_val_plot(xg, rf):
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    xg = xg * 0.95
    lst = [xg, rf]
    label = ["XG", "RF"]
    i = 0
    for df in lst:
        df.loc[df.r2 < 0] = 0

        axs[0, 0].plot(df.N, df.r2, label=label[i])

        i = i + 1
    axs[0, 0].set_xlabel("Stations Added", fontsize=12)
    axs[0, 0].set_ylabel("R2 Score", fontsize=12)
    axs[0, 0].legend()
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    i = 0
    for df in lst:
        df.loc[df.rms < 0] = 0

        axs[0, 1].plot(df.N, df.rms, label=label[i])

        i = i + 1
    axs[0, 1].set_xlabel("Stations Added", fontsize=12)
    axs[0, 1].set_ylabel("RMSE (mm)", fontsize=12)
    axs[0, 1].legend()
    i = 0
    for df in lst:
        df.loc[df.mae < 0] = 0

        axs[1, 0].plot(df.N, df.mae, label=label[i])

        i = i + 1
    axs[1, 0].set_xlabel("Stations Added", fontsize=12)
    axs[1, 0].set_ylabel("MAE (mm)", fontsize=12)
    axs[1, 0].legend()

    i = 0
    for df in lst:
        # df.loc[df.mbe < 0] = 0

        axs[1, 1].plot(df.N, df.mbe, label=label[i])

        i = i + 1
    axs[1, 1].set_xlabel("Stations Added", fontsize=12)
    axs[1, 1].set_ylabel("MBE (mm)", fontsize=12)
    axs[1, 1].legend()
    axs[0, 0].set_title("a")
    axs[0, 1].set_title("b")
    axs[1, 0].set_title("c")
    axs[1, 1].set_title("d")
    fig.suptitle("Training of XGB model on Alaskan data", fontsize=16)
    plt.show()


# Plot comparing itterations of xg/rf models
def it_plot(r_lst, lable, xg=False):
    fig, axs = plt.subplots(len(r_lst))
    fig.tight_layout()

    if xg:
        axs[2].set_xscale("log")
        r_lst[2].loc[r_lst[2].r2 < 0] = 0

    for i in range(len(r_lst)):
        axs[i].plot(r_lst[i].iloc[:, 5], r_lst[i].rms, "-bo", label="RMSE")
        axs[i].plot(r_lst[i].iloc[:, 5], r_lst[i].mae, "-r^", label="MAE")
        axs[i].plot(r_lst[i].iloc[:, 5], r_lst[i].r2 * 100, "-gs", label="R2")
        axs[i].grid()
        axs[i].set_xlabel(lable[i], fontsize=12)
        axs[i].legend()
    fig.suptitle("Training XGB model", fontsize=16)
    plt.show()


# Heavily tailored bar plots for each separate category
def bar_SWE(ax, df, type, ttl):
    if type == "SWE":
        X = np.arange(50, 2001, 50)
        lst = []
        for x in X:
            temp = df.loc[(df.SWE < x)]
            df = df.drop(temp.index)
            lst.append(temp.shape[0])
        lst.append(df.shape[0])
        labels = [str(x) for x in X]
        labels.append(">2000")
        labels[0] = "<50"
        Title = ttl
        vis = False
        xlabel = "SWE in mm"
    elif type == "SD":
        X = np.arange(10, 450, 10)
        lst = []
        for x in X:
            temp = df.loc[(df.SD < x)]
            df = df.drop(temp.index)
            lst.append(temp.shape[0])
        lst.append(df.shape[0])
        labels = [str(x) for x in X]
        labels.append(">450")
        labels[0] = "<10"
        Title = ttl
        vis = False
        xlabel = "SD in cm"
    elif type == "density":
        X = np.linspace(0, 0.7, 22)
        lst = []
        df["density"] = (df.SWE / 10) / df.SD
        for x in X:
            temp = df.loc[(df.density < x)]
            df = df.drop(temp.index)
            lst.append(temp.shape[0])
        lst.append(df.shape[0])
        labels = [str(x.round(2)) for x in X]
        for x, y in zip(lst, labels):
            print(y, x)
        labels.append(">0.7")
        labels[0] = "0"
        Title = ttl
        vis = True
        xlabel = "Density (g $\mathregular{cm^{-1}}$)"
    elif type == "elevations":
        X = np.arange(100, 3300, 200)
        lst = []
        for x in X:
            temp = df.loc[(df.Elev < x)]
            df = df.drop(temp.index)
            lst.append(temp.shape[0])
        lst.append(df.shape[0])
        labels = [str(x) for x in X]
        labels.append(">3300")
        labels[0] = "<100"
        Title = ttl
        vis = False
        xlabel = "Elevation in m"
    elif type == "stations":
        a = df["StationID"].unique()
        try:
            lst = pd.read_csv("Station_plot_AK.csv", index_col=0).values.reshape(-1)
            print("read Station plot from Csv")
        except:
            lst = []
            for x in a:
                temp = df.loc[(df.StationID == x)]
                df = df.drop(temp.index)
                lst.append(temp.shape[0])

            pd.DataFrame(lst).to_csv("Station_plot_AK.csv")
            print("Created Station Csv")

        labels = np.arange(len(lst))
        print(labels)
        Title = ttl
        vis = True
        xlabel = "Stations"
    elif type == "time":
        X = np.arange(7, 280, 7)

        lst = []
        for x in X:
            temp = df.loc[(df.DOY < x)]
            df = df.drop(temp.index)
            lst.append(temp.shape[0])
        lst.append(df.shape[0])
        labels = pd.date_range(start="2022-09-01", end="2023-06-08", periods=len(X))
        labels = [str(x.month) + "-" + str(x.day) for x in labels]
        labels.append(">06-08")

        Title = ttl
        vis = False
        xlabel = "Date"

    if type != "stations":
        ax.bar(
            labels,
            lst,
            align="center",
            # density=True,
            width=0.8,
            facecolor="gray",
            # alpha=0.5,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_xticklabels(labels, rotation="vertical")
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(vis)
    else:
        ax.bar(
            np.arange(len(lst)),
            lst,
            align="center",
            # density=True,
            width=0.8,
            facecolor="gray",
            # alpha=0.5,
            edgecolor="black",
            linewidth=1,
        )

    # ax.set_ylabel("Number of record")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(Title, fontsize=16, loc="left")


def records(df, val, ttl="USCN Dataset Distribution"):
    df = df.append(val)

    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    bar_SWE(axs[(0, 0)], df, type="SWE", ttl="a")
    bar_SWE(axs[(0, 1)], df, type="SD", ttl="b")
    bar_SWE(axs[(0, 2)], df, type="density", ttl="c")
    bar_SWE(axs[(1, 0)], df, type="time", ttl="d")
    bar_SWE(axs[(1, 1)], df, type="elevations", ttl="e")
    bar_SWE(axs[(1, 2)], df, type="stations", ttl="f")

    fig.text(
        0.04,
        0.5,
        "Number of records",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize="16",
    )
    fig.suptitle(ttl, fontsize=16)
    plt.savefig("Dataset_plots6.png")
    plt.show()


# Scatterplot of snow-density distribution by snowclass
def scatter_plot(df, val_df):
    df = df.append(val_df)
    df["density"] = (df.SWE.values / 10) / df.SD.values
    df["density"].loc[df.density > 0.7] = 0.7
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

    colors = ["blue", "gray", "green", "red", "brown", "yellow"]
    cnt = 0
    ci = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for i, c in zip(ci, classes):
        temp_df = df.loc[(df.snowclass == c)]
        print(
            temp_df.shape[0],
            len(list(temp_df["StationID"].unique())),
            temp_df.shape[0] * 100 / df.shape[0],
        )
        swe = temp_df.SWE
        sd = temp_df.SD
        den = temp_df.density

        axs[i].scatter(sd, den, c=colors[cnt], s=0.5)
        axs[i].set_ylabel("Density (g $\mathregular{cm^3}$)")
        axs[i].set_xlabel("Depth (cm)")
        axs[i].legend(c)
        cnt += 1
    fig.suptitle("Canada and US West Coast Density Scatter Plot by Class", fontsize=16)
    plt.show()

    fig, axs = plt.subplots(1)

    cnt = 0
    for i, c in zip(ci, classes):
        temp_df = df.loc[(df.snowclass == c)]

        swe = temp_df.SWE
        sd = temp_df.SD
        den = temp_df.density

        axs.scatter(sd, den, c=colors[cnt], s=2)
        axs.set_ylabel("Density (g $\mathregular{cm^3}$)")
        axs.set_xlabel("Depth (cm)")
        cnt += 1
    axs.legend(classes)
    fig.suptitle("Canada and US West Coast Density Scatter Plot", fontsize=16)
    plt.show()


# PDF Plot over a single dataset
def pdf_plot_single(df, Title="Probability density function for the Alaska Dataset"):
    xlabel = ["SWE (mm)", "SD (cm)", "Density (g $\mathregular{cm^3}$)"]
    ylabel = ["A)", "B)", "C)"]
    labels = ["Training"]

    df["density"] = (df.SWE.values / 10) / df.SD.values

    df["SWE"].loc[df.SWE > 2000] = 2001
    df["SD"].loc[df.SD > 450] = 451
    df["density"].loc[df.density > 0.7] = 0.7
    df["density"].loc[df.density < 0.05] = 0.05
    swe = df.SWE
    sd = df.SD
    den = df.density

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(Title, fontsize=16)

    for i in range(3):
        axs[i].hist(swe, 50, density=True, facecolor="gray", alpha=1)
        axs[i].set_xlabel(xlabel[i], fontsize=12)
        axs[i].set_title(ylabel[i], fontsize=12, loc="left")
        axs[i].legend(labels)

    fig.text(
        0.04,
        0.5,
        "Probability density function (pdf)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize="12",
    )
    plt.show()


# Plot to compare PDF of training and validaion datasets compared
def pdf_plot_comparing(
    df,
    val_df,
    Title="Probability density function for training and validation datasets",
):
    xlabel = ["SWE (mm)", "SD (cm)", "Density (g $\mathregular{cm^3}$)"]
    ylabel = ["A)", "B)", "C)"]

    # Calculating density
    df["density"] = (df.SWE.values / 10) / df.SD.values
    val_df["density"] = (val_df.SWE.values / 10) / val_df.SD.values

    # Merging outliers to get the best plot
    df["SWE"].loc[df.SWE > 2000] = 2001
    val_df["SWE"].loc[val_df.SWE > 2000] = 2001

    df["SD"].loc[df.SD > 450] = 451
    val_df["SD"].loc[val_df.SD > 450] = 451

    df["density"].loc[df.density > 0.7] = 0.7
    val_df["density"].loc[val_df.density > 0.7] = 0.7

    df["density"].loc[df.density < 0.05] = 0.05
    val_df["density"].loc[val_df.density < 0.05] = 0.05

    swe = df.SWE
    sd = df.SD
    den = df.density

    swe_v = val_df.SWE
    sd_v = val_df.SD
    den_v = val_df.density

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(Title, fontsize=16)
    handles = [
        Rectangle((0, 0), 1, 1, color=c, ec=e)
        for c, e in zip(["gray", "none"], ["none", "black"])
    ]
    labels = ["Training", "Validation"]
    for i in range(3):
        axs[i].hist(swe, 50, density=True, facecolor="gray", alpha=0.5)
        axs[i].hist(
            swe_v,
            50,
            density=True,
            facecolor="none",
            alpha=1,
            edgecolor="black",
            linewidth=1.5,
        )
        axs[i].set_xlabel(xlabel[i], fontsize=12)
        axs[i].set_title(ylabel[i], fontsize=12, loc="left")

        axs[i].legend(handles, labels)
    fig.text(
        0.04,
        0.5,
        "Probability density function (pdf)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize="12",
    )
    plt.show()


# Creates heatmap using predicted and measured values, smooted by a
# factor of s
def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


# Has to be tailored to best illustrate the results
def heatmap_results(
    df_lst, figtitle_lst, ttl="Heatmaps of model performance", AK=False
):

    fig, axs = plt.subplots(nrows=3, ncols=5, sharey="row")
    ii = [0, 1, 2, 3, 4]
    ji = [0, 1, 2]
    for df, i, figtitle in zip(df_lst, ii, figtitle_lst):
        depth = 350
        if AK:
            depth = 180
        sdf = df.loc[df.SD > depth]

        print(sdf.shape)
        df = df.loc[df.SD < depth]
        if sdf.shape[0] > 1:
            sdf["SD"] = sdf.SD * 10 / max(sdf.SD) + depth
            df = df.append(sdf)
        df["density"] = (df.SWE.values / 10) / df.SD.values
        df = df.drop(df.loc[df.density > 0.7].index)
        df = df.drop(df.loc[df.density < 0.005].index)
        print("wee")
        swe_p = df.Predicted
        swe = df.SWE
        sd = df.SD
        den = df.density

        den_y = (swe_p / 10) / sd

        error_df = df

        error_df["error"] = swe.values - swe_p.values

        error_df["error_r"] = (error_df.error / swe.values) * 100
        error_df["error_d"] = den.values - den_y

        error_df = error_df.loc[(error_df.error > -600)]
        error_df = error_df.loc[(error_df.error < 600)]
        error_df = error_df.loc[(error_df.error_r < 60)]
        error_df = error_df.loc[(error_df.error_r > -60)]
        print(error_df.shape, "SHAPE")
        x = error_df.SD.values

        err_lst = [
            error_df.error.values,
            error_df.error_r.values,
            error_df.error_d.values,
        ]
        x_lst = [error_df.SD.values, error_df.SD.values, error_df.density.values]

        # Generate some test data
        y = error_df.error_r.values
        sigmas = 40
        title_lst = ["A)", "B)", "C)"]
        ylabel_lst = [
            "Error in SWE (mm)",
            "Error in SWE (%)",
            "Error in density (g $\mathregular{cm^3}$)",
        ]

        for j, title, y, ylabel in zip(ji, title_lst, err_lst, ylabel_lst):
            print(j)
            img, extent = myplot(error_df.SD.values, y, sigmas)

            axs[j, i].imshow(
                img, extent=extent, origin="lower", cmap=cm.jet, aspect="auto"
            )
            CS = axs[j, i].contour(img, extent=extent, origin="lower", colors="black")
            axs[j, i].clabel(CS, inline=1, fontsize=10)
            # axs[j, i].set(ylabel=ylabel, xlabel="Depth (cm)")
            axs[j, i].set_facecolor("xkcd:darkblue")
            axs[j, i].set_title(str(i + 1) + title, fontsize=12, loc="left")
        # fig.suptitle(figtitle)

    for ax, col in zip(axs[0], figtitle_lst):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, 5),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    rows = [
        "Error in SWE (mm)",
        "Error in SWE (%)",
        "Error in density (g $\mathregular{cm^-3}$)",
    ]
    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation="vertical",
        )

    fig.text(
        0.5,
        0.04,
        "Depth (cm)",
        va="center",
        ha="center",
        fontsize="12",
    )
    fig.suptitle(ttl, fontsize=16)
    plt.show()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def box_whisk(models, names):

    fig, axs = plt.subplots(1, len(models), sharey=True)
    fig.suptitle("Distribution of Residuals, Alaska", fontsize=14)
    axs[0].set_ylabel("Error SWE in mm", fontsize=12)
    for m, ms, ax in zip(models, names, axs):
        error = pd.DataFrame()
        error["Error"] = m.SWE - m.Predicted

        ax.boxplot(error.Error, showfliers=False, whis=[5, 95])
        ax.set_title(ms)

        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid()
        ax.set_xticks([2])
        # ax.set_xticklabels(ms)

        q = plt.cbook.boxplot_stats(error.Error, whis=[5, 95])
        quantiles = np.array(
            [
                q[0]["whislo"],
                q[0]["q1"],
                q[0]["med"],
                q[0]["q3"],
                q[0]["whishi"],
            ]
        )
        print(np.quantile(error.Error, np.array([0.05, 0.25, 0.50, 0.75, 0.95])))
        print(quantiles)

    plt.show()


def scatter_plot(df_lst, names, suffix):
    fig, axs = plt.subplots(2, 3)
    # print(df_lst)
    ldf = pd.DataFrame(columns=["mse", "rms", "mae", "mbe", "r2", "Name"])
    for df, name, ax in zip(df_lst, names, axs.flat):

        y = df.SWE
        x = df.Predicted

        ax.plot(y, x, "ok", markersize=3, alpha=0.2)

        ax.grid()
        lims = [0, 2000]  # min of both axes  # max of both axes

        # now plot both limits against eachother
        ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, alpha=0.55, c="red")

        ax.set_xlabel("Predicted SWE (mm)")
        ax.set_ylabel("Measured SWE (mm)")
        ax.set_title(name)
        mse, rms, mae, mbe, r2 = accuracy_score(y, x)

        ldf.loc[ldf.shape[0]] = [mse, rms, mae, mbe, r2, name]

    fig.savefig("AK_SCATTER_" + name + ".png", dpi=1000)
    ldf.to_csv("Results/Combined_results_" + suffix + ".csv")
    plt.show()


def Scatter_hist(df, fig):
    x = df.SWE
    y = df.Predicted
    # fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    # scatter points on the main axes
    main_ax.plot(x, y, "ok", markersize=3, alpha=0.2)
    main_ax.plot([0, 1], [0, 1], transform=ax.transAxes, c="red")
    main_ax.grid()
    # histogram on the attached axes
    x_hist.hist(x, 100, histtype="stepfilled", orientation="vertical", color="gray")
    x_hist.invert_yaxis()

    y_hist.hist(y, 100, histtype="stepfilled", orientation="horizontal", color="gray")
    y_hist.invert_xaxis()
    return fig


lable_it_xg = ["Number of estimators", "Number of layors", "Learning rate"]
lable_it_rf = ["Number of estimators", "Number of layors", "Number of Leaves"]
names = ["XGBoost", "Random Forest", "MLP", "Jonas", "Sturm"]


def feat_an(df, r2, cols):
    ccols = [
        "SD",
        "Elev",
        "DOY",
        "D without S",
        "n frost-defrost",
        "a-+ degrees",
        "avg age Scover",
        "n layer",
        "a-solid P",
        "a-solid P last 10D",
        "T P last 10D",
        "a-temp last 6D",
        "snowclass",
    ]
    df.columns = cols
    fig, axs = plt.subplots()
    fig.tight_layout()

    for col in df.columns:
        axs.plot(df.index, df[col], label=col)

    axs.set_xlabel("Number of Estimators", fontsize=12)
    axs.set_ylabel("Normalized Frequency score", fontsize=12)

    ax2 = axs.twinx()
    ax2.plot(df.index, r2.r2, "--", label="R2 Score Alaska")
    ax2.plot(df.index, r2.xr2, "--", label="R2 Score USCN")
    ax2.legend()
    ax2.set_ylabel("R2 Score", fontsize=12)
    axs.legend()
    axs.grid()

    fig.suptitle("Feature Frequency as a Function of Estimators", fontsize=14)
    plt.show()


def run_feat_xg():
    feature_columns_to_use = [
        "SD",
        "Elev",
        # "Lat",
        # "Long",
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
        "snowclass",
    ]
    feat_anal = pd.read_csv("feat_progression_score.csv")
    feat_anal = feat_anal.fillna(0)

    feat_r2 = pd.read_csv("IT_RES/feat_score_xg_results_N_est_D7.csv")
    feat_r2[feat_r2 < 0] = 0
    feat_r2.index += 1

    feat_anal = feat_anal[feature_columns_to_use]
    feat_anal = feat_anal.div(feat_anal.sum(axis=1), axis=0)

    feat_anal.index += 1

    feat_an(feat_anal, feat_r2, feature_columns_to_use)


# Tailored MLP-plotting scripts.
def mlp_param_USCN(classes):
    lst = []
    for c in classes:
        df = pd.read_csv("MLP_val_res" + c + ".csv")
        lst.append(df)

    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    layers = [2, 3, 4, 5, 6, 10, 25]
    ep = [10, 25, 50, 75, 100]
    i_ = [1, 2, 3, 4, 5, 6, 7]
    for df, ax, c in zip(lst, axs.flat, classes):

        for L in layers:
            adf = df.loc[df.Hid == L]
            ax.plot(i_, adf.r2, label=str(L))
            ax.grid(b=True, which="major", color="gray", linestyle="--")
            ax.xaxis.set_ticklabels(ep)
            ax.set_title(c)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("R2 Score")
            ax.legend(title="Hidden Layers")
    plt.show()


def mlp_param_AK(classes):
    lst = []
    for c in classes:
        df = pd.read_csv("MLP_val_res_AK2" + c + ".csv")
        df = df.fillna(0)
        lst.append(df)

    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    layers = [2, 3, 4, 5, 6, 10, 25]
    ep = [10, 25, 50, 75, 100]
    i_ = [1, 2, 3, 4, 5]
    for df, ax, c in zip(lst, axs.flat, classes):

        for L in layers:
            adf = df.loc[df.Hid == L]
            print(adf)
            ax.plot(adf.Ep, adf.r2, label=L)
            ax.grid(b=True, which="major", color="gray", linestyle="--")
            # ax.xaxis.set_ticklabels()
            ax.set_xlabel("Epochs", fontsize=12)
            ax.set_ylabel("R2 Score", fontsize=12)
            ax.set_title("Parameter Tuning of Single MLP using Alaskan Validation")

            ax.legend(title="Hidden Layers:", loc="lower right")
    plt.show()


def station_cross_validation():
    x_label = "Number of Estimators"
    y_label = ["R2 Score", "RMSE (mm)", "MAE (mm)", "MBE (mm)"]
    legend = "State"
    title = ["a", "b", "c", "d"]

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()

    lst = []
    ilst = ["CA", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"]
    for i in ilst:
        df = pd.read_csv("IT_RES/" + i + "_score_xg_results_N_est_D7.csv")
        lst.append(df)

    for ax in axs.flat:
        for df, i in zip(lst, ilst):
            df.loc[df.r2 < 0] = 0

            axs[0, 0].plot(df.N, df.r2, label=str(i))

            # Code to mark the highest point on the graph
            x = df.N
            y = df.r2
            xmax = x[np.argmax(y)]
            ymax = y.max()

            text = ""
            tform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            arrowprops = dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60"
            )
            kw = dict(
                xycoords=tform,
                textcoords=tform,
                # bbox=bbox_props,
                ha="right",
                va="top",
            )
            ax.scatter(xmax, ymax)
            ax.annotate(
                text,
                xy=(xmax + 0.0, ymax + 0.0),
                xytext=(xmax + 0.0, ymax + 0.00),
                **kw
            )
        break

    for df, i in zip(lst, ilst):
        adf = df.loc[df.rms < 0] = 0

        axs[0, 1].plot(adf.N, adf.rms, label=str(i))
        adf = df.loc[df.mae < 0] = 0

        axs[1, 0].plot(adf.N, adf.mae, label=str(i))
        axs[1, 1].plot(df.N, df.mbe, label=str(i))

    j = 0
    for x in range(2):
        for y in range(2):
            axs[x, y].set_xlabel(x_label, fontsize=12)
            axs[x, y].set_xlabel(y_label[j], fontsize=12)
            axs[x, y].legend(legend)
            axs[x, y].set_title(title[j])
            axs[x, y].grid()

    plt.show()
    fig.suptitle("Block Bootstrap results from Additional US States", fontsize=14)

    plt.show()


def xg_it_stations():
    x_label = "Number of Estimators"
    y_label = ["R2 Score", "RMSE (mm)", "MAE (mm)", "MBE (mm)"]
    legend = "Stations Added:"
    title = ["a", "b", "c", "d"]

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()

    lst = []
    for i in range(0, 101, 20):
        df = pd.read_csv("IT_RES/AK_stations/xg_results_ALASKA_IT_" + str(i) + ".csv")
        lst.append(df)
    i = 0
    for df in lst:
        df.loc[df.r2 < 0] = 0
        axs[0, 0].plot(df.Nt, df.r2, label=str(i))

        df.loc[df.rms < 0] = 0
        axs[0, 1].plot(df.Nt, df.rms, label=str(i))
        df.loc[df.mae < 0] = 0

        axs[1, 0].plot(df.Nt, df.mae, label=str(i))
        axs[1, 1].plot(df.Nt, df.mbe, label=str(i))
        i = i + 20

    j = 0
    for x in range(2):
        for y in range(2):
            axs[x, y].set_xlabel(x_label, fontsize=12)
            axs[x, y].set_xlabel(y_label[j], fontsize=12)
            axs[x, y].legend(legend)
            axs[x, y].set_title(title[j])
            axs[x, y].grid()
    plt.show()


xg_it_stations()