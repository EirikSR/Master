# load Canadian historical sonw survey (CHSS): dictionary with Station ID,
#  longitude, latitude, elevation, 2 dim matrix (Station ID x dates) for
# SWE, SD, density
import pandas as pd
import datetime
import numpy as np
import pickle


def feature_calculations(SWE, SD, total_precip, tmin, tmax, Stations, savefile):

    # correction of temperature because of different elevation of station and grid
    # load elevation of grid points
    elevation = Stations["ERA5_Elev"]
    # round Longitude and Latitude to the first decimal to fit them together (ERA5 lowest resolution is 0.1x0.1)

    Lat_station = Stations["Lat"].round(decimals=1)
    Long_station = Stations["Lon"].round(decimals=1)
    # apply temperature correction
    for i in Lat_station.index:
        elev_grid = elevation["Elev"].loc[i]
        elev_st = elevation["Elev"].loc[i]
        tmin.iloc[i, :] = (
            tmin.iloc[i, :] + ((elev_st - elev_grid) / 1000 * (-6)) - 272.15
        )
        tmax.iloc[i, :] = (
            tmax.iloc[i, :] + ((elev_st - elev_grid) / 1000 * (-6)) - 272.15
        )

    # load function to calculate input varibles from meteorological data
    import _input_calc
    import pandas as pd
    from datetime import datetime, timedelta

    # create dataset for input;
    # one line represents one records of SD and the associated input varibles, which will be calculated in the following
    MLP_data = pd.DataFrame(
        columns=[
            "StationID",
            "Date",
            "SD",
            "SWE",
            "Elev",
            "Lat",
            "Long",
            "Day of year",
            "days without snow",
            "number frost-defrost",
            "accum pos degrees",
            "average age SC",
            "number layer",
            "accum solid precip",
            "accum solid precip in last 10 days",
            "total precip last 10 days",
            "average temp last 6 days",
        ]
    )

    start_year = pd.to_datetime(tmin.columns[0]).year
    end_year = pd.to_datetime(tmin.columns[-1]).year
    range_year = np.arange(start_year + 1, end_year + 1)
    dates_Meteo = pd.to_datetime(tmin.columns)
    nb_dates = len(dates_Meteo)

    # count to print how may stations are left
    count = len(Stations.index)
    line = 0
    for St in Stations.index:
        count = count - 1
        if count % 25 == 0:
            print(datetime.now())
            print("{} left till loop over stations done".format(count))
        # calculate average temeperature
        tmid = (tmin.iloc[St, :] + tmax.iloc[St, :]) / 2

        # logistic regression to separate precipitation into solid and liquid parts
        # tested over the northern hemnisphere by
        # ('Spatial variation of the rain–snow temperature threshold across the Northern Hemisphere' Jennings et al. 2018)
        probab_snow = 1 / (1 + np.exp(-1.54 + 1.24 * tmid))
        total_precip_solid = total_precip.iloc[St, :] * probab_snow

        # generate frost-defrost timeline for concerning station;
        # -1°C for the maximal temp and 1°C for the minimal temperature are the threshold for freezing and thawing, respectively
        frost_defrost_vect = _input_calc.frost_defrost(
            tmin.iloc[St, :], tmax.iloc[St, :], dates_Meteo, nb_dates
        )
        # print(tmin.iloc[St, 0])
        # generate timeline of number of days without snow for concerning station
        nb_days_without_snow_vect = _input_calc.num_without_snow(
            tmax.iloc[St, :], total_precip.iloc[St, 1:], dates_Meteo, nb_dates
        )
        # generate timeline of accumulated posiitve degrees since beginning of winter (1. of September)
        pos_degree_vect = _input_calc.pos_degrees(tmid, dates_Meteo, nb_dates)
        # generate timeline of number of layers;
        # a new layer is considered to be created if there is a 3-days gap
        # with less than 10mm of (accumulated over these 3 days) solid precipitation
        num_layer_vec = _input_calc.num_layer(
            3, 10, total_precip_solid, dates_Meteo, nb_dates
        )
        print(SWE)
        # loop over the dates for the concerning station
        for dat in SWE.columns:
            if dat != "StationID":

                if ~np.isnan(SWE.loc[St, dat]):
                    # preparation
                    ndRow = np.empty((1, len(MLP_data.columns)))
                    ndRow[:] = np.nan
                    row = pd.DataFrame(ndRow, columns=MLP_data.columns)
                    MLP_data = MLP_data.append(row, ignore_index=True)

                    # add Station ID
                    MLP_data.iloc[line]["StationID"] = Stations["StationID"][St]
                    # add Lat and Long
                    MLP_data.iloc[line]["Lat"] = Stations.loc[St, "Lat"]
                    MLP_data.iloc[line]["Long"] = Stations.loc[St, "Lon"]
                    # add date of measurement
                    MLP_data.iloc[line]["Date"] = dat
                    # add station elevation
                    MLP_data.iloc[line]["Elev"] = Stations.loc[St, "Elev"]
                    # add snow bulk denisty
                    # MLP_data.iloc[line]['Den'] = CHSS['Den(kg/m3)'].iloc[St, dat]
                    # add SWE
                    MLP_data.iloc[line]["SWE"] = SWE.loc[St, dat]
                    # add snow depth
                    MLP_data.iloc[line]["SD"] = SD.loc[St, dat]
                    # days without snow since 1st of august
                    MLP_data.iloc[line][
                        "days without snow"
                    ] = nb_days_without_snow_vect.loc[dat][0]
                    # print(nb_days_without_snow_vect)
                    # number of frost-defrost events since 1st of September
                    MLP_data.iloc[line][
                        "number frost-defrost"
                    ] = frost_defrost_vect.loc[dat, 0]
                    # calculate the accumulated temperature from 1st of September till record
                    MLP_data.iloc[line]["accum pos degrees"] = pos_degree_vect.loc[dat][
                        0
                    ]
                    # calculate the average age of the snow cover
                    (
                        MLP_data.iloc[line]["average age SC"],
                        total_precip_mod_solid,
                        cumul,
                        nb_days,
                    ) = _input_calc.age_snow_cover(dat, tmid, total_precip_solid)
                    # add number of days since 1st of September
                    MLP_data.iloc[line]["Day of year"] = nb_days
                    # estimate the number of layers in the snow cover
                    MLP_data.iloc[line]["number layer"] = num_layer_vec.loc[dat][0]
                    # calculate accumlated solid precipitation from 1st September till record
                    MLP_data.iloc[line]["accum solid precip"] = cumul
                    # calculate accumlated solid precipitation in the last 10 days before the record
                    MLP_data.iloc[line]["accum solid precip in last 10 days"] = np.sum(
                        total_precip_mod_solid[-10:]
                    )
                    # calculate accumlated total precipitation in the last 10 days before the record
                    dat_Ndays = pd.to_datetime(dat) - timedelta(days=10)
                    dat_Ndays = (
                        str(dat_Ndays.year)
                        + "-"
                        + "{:02d}".format(dat_Ndays.month)
                        + "-"
                        + "{:02d}".format(dat_Ndays.day)
                    )
                    MLP_data.iloc[line]["total precip last 10 days"] = np.sum(
                        total_precip.loc[St, dat:dat_Ndays:-1]
                    )
                    # calculate average temperature in the last 6 days before the record
                    dat_Ndays = pd.to_datetime(dat) - timedelta(days=6)
                    dat_Ndays = (
                        str(dat_Ndays.year)
                        + "-"
                        + "{:02d}".format(dat_Ndays.month)
                        + "-"
                        + "{:02d}".format(dat_Ndays.day)
                    )

                    MLP_data.iloc[line]["average temp last 6 days"] = np.mean(
                        tmid.loc[dat:dat_Ndays:-1]
                    )
                    # increase line count
                    line += 1

    MLP_data.to_csv(savefile, index=False)