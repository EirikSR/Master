import pandas as pd
import csv
import numpy as np
import requests as req
import re
from os import stat
import xarray as xr
import cfgrib
from os.path import exists
from netCDF4 import Dataset

# State abreviations for states with snotel data
states = ["AZ", "AK", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "SD", "UT", "WA", "WY"]


def get(url, params={}, output=None):
    """Takes a url ar well as optional wikipedia parameters and returnd a regex object containgin a string version of the html

    Args:
        url (str)       : A String of a wikipedia url
        params (str)    : (Optional) Wikipedia parameters, will change regex output
        output (str)    : (Optional) If passed, the html is saved to a file with $output as name

    Writes:
        html            :(Optional) Writes string from regex object to file if filname is passed

    Returns:
        regex object(obj): A regex object containing the html code as a string
    """
    r = req.get(url, params=params)

    if output != None:
        HTML_file = open(output, "w", encoding="utf-8")
        HTML_file.write("<!-- " + r.url + "--> \n")
        HTML_file.write(r.text)
        HTML_file.close()

    # print(r.url)
    return r


def find_urls(regex, output=None):
    """Takes a string of a regex object or string containing html code, extracts all urls found in the article and

    Args:
        regex (obj)/(str): A String consisting of html code (from a wikipedia article) or a regex object containing a string of html
        Output(str)      : (Optional) Of passed, saved all urls to a file with name $output
    Writes:
        list            : (Optional) Is output variable is passed, list will be written
    Returns:
        list             : List of the all urls found in article without duplicates
    """
    r = regex
    # Determines wether a string or object is passed, creates a list of possible urls
    if type(r) == str:

        pat = r"(?<=href\=\").+?(?=\"|\#|\ t)"
        urls = re.findall(
            pat,
            r,
        )
    else:
        pat = r"(?<=href\=\").+?(?=\"|\#|\ t)"
        urls = re.findall(
            pat,
            r.text,
        )

    # Reject urls were used for toubleshooting and locating missed formats
    lst = []
    reject = []

    # For each url, tries to reconstruct links by comparing, then appending to string. Then appenging fixed link to output list. If not reconstructable, added to rejects
    for u in urls:

        if u[-5:] == "_hist":

            u = "https://wcc.sc.egov.usda.gov" + u
            # Partitioning the url is done in two steps due to http: and https: both containing ':'
            head, sep, tail = u.partition(":")
            head2, sep2, tail = tail.partition(":")
            string = head + sep + head2
            lst.append(string)

    urls = list(dict.fromkeys(lst))

    # Writes links to text file
    if output != None:
        with open(output, "w") as f:
            for x in urls:
                f.write(x + "\n")
        f.close()

    return lst


def scrap_data():
    mont_dict = {
        "Dec": "-12-",
        "Jan": "-01-",
        "Feb": "-02-",
        "Mar": "-03-",
        "Apr": "-04-",
        "May": "-05-",
        "Jun": "-06-",
        "Jul": "-07-",
        "Aug": "-08-",
        "Sep": "-08-",
    }

    # Base url
    url = "https://wcc.sc.egov.usda.gov/nwcc/rgrpt?report=snowmonth_hist&state="
    for state in states:
        r = get(url + state)
        urls = find_urls(r)
        print("state")

        t = get(urls[0])

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, "html.parser")
        soup_table = soup.find_all("table")

        df = pd.read_html(str(soup_table[1]))[0]
        print(df.shape, len(urls))
        print([df.iloc[1, 4]])
        col_names = [
            "Year",
            "Jan_date",
            "Jan_SD",
            "Jan_SWE",
            "Feb_date",
            "Feb_SD",
            "Feb_SWE",
            "Mar_date",
            "Mar_SD",
            "Mar_SWE",
            "Apr_date",
            "Apr_SD",
            "Apr_SWE",
            "May_date",
            "May_SD",
            "May_SWE",
            "Jun_date",
            "Jun_SD",
            "Jun_SWE",
        ]
        super_lst = [["Date", "StationID", "Lat", "Long", "Elevation", "SD", "SWE"]]
        for i in range(len(urls)):

            print(state, i)
            t = get(urls[i])

            # Using B-Soup to create array of data in html file

            soup = BeautifulSoup(t.text, "html.parser")
            li = soup.prettify().split("\n")
            # The data starts after 62 lines
            li = li[62:]

            lif = []
            for l in li:
                lif.append(l.split(","))

            # Converting data table to pandas dataframe
            m_df = pd.DataFrame(lif)

            # Rather ugly regex-code, copy-pase code could be removed.
            try:
                m_df.columns = col_names
                # print(m_df)
                for index, m in m_df.iterrows():
                    # print(m.Year)
                    if m.Year != "":
                        if 2018 > float(m.Year) > 1980:
                            if m.Jan_SD != "" and m.Jan_SWE != "":
                                if float(m.Jan_SD) > 10 and float(m.Jan_SWE) > 0:
                                    yr = m.Year

                                    # Where no date is suplied, first day in month is assumed
                                    if m.Jan_date == "":
                                        d = "-01-01"
                                    else:
                                        mnt = mont_dict[m.Jan_date[0:3]]
                                        d = mnt + m.Jan_date[-2:]
                                        # Some december measurements are included in following year
                                        if m.Jan_date[0:3] == "Dec":
                                            yr = int(yr) - 1
                                    super_lst.append(
                                        [
                                            str(yr) + d,
                                            df.iloc[i, 2],
                                            df.iloc[i, 5],
                                            df.iloc[i, 6],
                                            df.iloc[i, 4],
                                            m.Jan_SD,
                                            m.Jan_SWE,
                                        ]
                                    )
                            if m.Feb_SD != "" and m.Feb_SWE != "":
                                if float(m.Feb_SD) > 10 and float(m.Feb_SWE) > 0:
                                    if m.Feb_date == "":
                                        d = "-02-01"
                                    else:
                                        mnt = mont_dict[m.Feb_date[0:3]]
                                        d = mnt + m.Feb_date[-2:]

                                        super_lst.append(
                                            [
                                                m.Year + d,
                                                df.iloc[i, 2],
                                                df.iloc[i, 5],
                                                df.iloc[i, 6],
                                                df.iloc[i, 4],
                                                m.Feb_SD,
                                                m.Feb_SWE,
                                            ]
                                        )
                            if m.Mar_SD != "" and m.Mar_SWE != "":
                                if float(m.Mar_SD) > 10 and float(m.Mar_SWE) > 0:
                                    if m.Mar_date == "":
                                        d = "-03-01"
                                    else:
                                        mnt = mont_dict[m.Mar_date[0:3]]
                                        d = mnt + m.Mar_date[-2:]

                                    super_lst.append(
                                        [
                                            m.Year + d,
                                            df.iloc[i, 2],
                                            df.iloc[i, 5],
                                            df.iloc[i, 6],
                                            df.iloc[i, 4],
                                            m.Mar_SD,
                                            m.Mar_SWE,
                                        ]
                                    )
                            if m.May_SD != "" and m.May_SWE != "":
                                if float(m.May_SD) > 10 and float(m.May_SWE) > 0:
                                    if m.May_date == "":
                                        d = "-05-01"
                                    else:
                                        mnt = mont_dict[m.May_date[0:3]]
                                        d = mnt + m.May_date[-2:]

                                    super_lst.append(
                                        [
                                            m.Year + d,
                                            df.iloc[i, 2],
                                            df.iloc[i, 5],
                                            df.iloc[i, 6],
                                            df.iloc[i, 4],
                                            m.May_SD,
                                            m.May_SWE,
                                        ]
                                    )
                            if m.Apr_SD != "" and m.Apr_SWE != "":
                                if float(m.Apr_SD) > 10 and float(m.Apr_SWE) > 0:
                                    if m.Apr_date == "":
                                        d = "-04-01"
                                    else:
                                        mnt = mont_dict[m.Apr_date[0:3]]
                                        d = mnt + m.Apr_date[-2:]

                                    super_lst.append(
                                        [
                                            m.Year + d,
                                            df.iloc[i, 2],
                                            df.iloc[i, 5],
                                            df.iloc[i, 6],
                                            df.iloc[i, 4],
                                            m.Apr_SD,
                                            m.Apr_SWE,
                                        ]
                                    )
                            if m.Jun_SD != "" and m.Jun_SWE != "":
                                if float(m.Jun_SD) > 10 and float(m.Jun_SWE) > 0:

                                    if m.Jun_date == "":
                                        d = "-06-01"
                                    else:
                                        mnt = mont_dict[m.Jun_date[0:3]]
                                        d = mnt + m.Jun_date[-2:]
                                    super_lst.append(
                                        [
                                            m.Year + d,
                                            df.iloc[i, 2],
                                            df.iloc[i, 5],
                                            df.iloc[i, 6],
                                            df.iloc[i, 4],
                                            m.Jun_SD,
                                            m.Jun_SWE,
                                        ]
                                    )
            # Print error message if unsuccesful
            except:
                print(str(i) + " Didnt work" + ", " + str(df.iloc[i, 2]))

        # Create seperate files for each state
        file = open("SNOTEL_" + state + ".csv", "w+", newline="")
        with file:
            write = csv.writer(file)
            write.writerows(super_lst)
        file.close()


def combine_station_data():

    # Loading dataframes for each state
    AZ_df = pd.read_csv("SNOTEL_AZ.csv")
    AK_df = pd.read_csv("SNOTEL_AK.csv")
    CA_df = pd.read_csv("SNOTEL_CA.csv")
    CO_df = pd.read_csv("SNOTEL_CO.csv")
    ID_df = pd.read_csv("SNOTEL_ID.csv")
    MT_df = pd.read_csv("SNOTEL_MT.csv")
    NM_df = pd.read_csv("SNOTEL_NM.csv")
    NV_df = pd.read_csv("SNOTEL_NV.csv")
    OR_df = pd.read_csv("SNOTEL_OR.csv")
    SD_df = pd.read_csv("SNOTEL_SD.csv")
    UT_df = pd.read_csv("SNOTEL_UT.csv")
    WA_df = pd.read_csv("SNOTEL_WA.csv")
    WY_df = pd.read_csv("SNOTEL_WY.csv")

    lst = [
        AZ_df,
        AK_df,
        CA_df,
        CO_df,
        ID_df,
        MT_df,
        NM_df,
        NV_df,
        OR_df,
        SD_df,
        UT_df,
        WA_df,
        WY_df,
    ]
    # Creating timeseries for covered time span
    ranger = pd.date_range(start="1980.01.01", end="2018.07.01")

    st_lst = []
    elev = []
    lat = []
    lon = []
    SD_df = pd.DataFrame(index=st_lst, columns=ranger)
    SWE_df = pd.DataFrame(index=st_lst, columns=ranger)

    # Creating a list of station meta-data. (coordinates and elevation)
    for df in lst:
        for index, row in df.iterrows():
            if row.StationID in st_lst:
                pass
            else:
                st_lst.append(row.StationID)
                elev.append(row.Elevation)
                lat.append(row.Lat)
                lon.append(row.Long)

    # Saving station data
    st_inf = pd.DataFrame(st_lst)
    elev_inf = pd.DataFrame(elev)
    lat_inf = pd.DataFrame(lat)
    lon_inf = pd.DataFrame(lon)

    station_df = pd.concat([st_inf, elev_inf, lat_inf, lon_inf], axis=1)
    station_df.columns = ["StationID", "Elev", "Lat", "Lon"]
    station_df.set_index("StationID", inplace=True)

    station_df.to_csv("USA_Stations.csv")

    # Creating two matrixes for SD and SWE, dates as columns and stations as rows.
    for df in lst:

        for index, row in df.iterrows():
            SWE_df[str(row.Date)][row.StationID] = row.SWE
            SD_df[str(row.Date)][row.StationID] = row.SD

    # Saving the matrixes for further proccesing
    SWE_df.to_csv("USA_SWE.csv")
    SD_df.to_csv("USA_SD.csv")


def order_era5():
    import cdsapi

    """
    Retreiving ERA5 data requires a key and precise parameters. This is only meant to serve 
    as an example of how a request is supposed to look like. For me, era5 was only able
    to give 10 years of data at a time. Precipitation data and temperature data are optained seperatly.
    Read https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form for 
    further information on era5-land data download and key registration,
    """
    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-land",
        {
            "format": "netcdf",
            "variable": "total_precipitation",
            "year": [
                "1980",
                "1981",
                "1982",
                "1983",
                "1984",
                "1985",
                "1986",
                "1987",
                "1988",
                "1989",
            ],
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "area": [
                49,
                -124,
                32,
                -103,
            ],
        },
        "download_total_precip_1980-1989.nc",
    )


def era5_proccessing(stations, file, temp=True):
    """Processing of era5-land dataset, assuming 10 year intervals of era5-land data in netcdf format.
    For the script to work, ecmwflibs needs to be installed through 'pip install ecmwflibs'
    """
    try:
        df = xr.open_dataset(file, engine="netcdf4")
    except:
        print("Could not read file using netcdf4")
        return 0

    if temp == True:

        saveloc = "Data/temp/1980/"
        # Printing xarray metadata
        for v in df:
            print(
                "{}, {}, {}".format(v, df[v].attrs["long_name"], df[v].attrs["units"])
            )

        # Finding start and stop time
        t0 = df.t2m[0, 0, 0].time.values
        t1 = df.t2m[-1, 0, 0].time.values
        # Creating timeseries of dates
        dates = pd.date_range(t0, t1, freq="d")

        for index, row in stations.iterrows():

            # Finding temperature using coordinates as index in xarray-df
            temp_df = df.t2m.sel(
                latitude=row.Lat, longitude=row.Lon, method="nearest"
            ).to_dataset()

            # Creating array of temperatures, reshaping so each row/col contains one list of 24 values
            # These values are hourly temperatures
            arr = temp_df.t2m.values
            arr = arr.reshape(-1, 24)

            # Calculating minimum and maximum temperature for each day, and storing separately
            min_ = np.min(arr, 1)
            max_ = np.max(arr, 1)

            # For each station, a file is created containing the date, min and max temperature.
            sdf = pd.DataFrame([dates, min_, max_]).T
            sdf.columns = ["time", "tmin", "tmax"]

            sdf.to_csv(saveloc + row.StationID + "_temp.csv", index=False)
            # These will have to be combined later.
    else:

        saveloc = "Data/tp/1980/"
        # Printing xarray metadata
        for v in df:
            print(
                "{}, {}, {}".format(v, df[v].attrs["long_name"], df[v].attrs["units"])
            )

        # Finding start and stop time
        t0 = df.tp[0, 0, 0].time.values
        t1 = df.tp[-1, 0, 0].time.values
        # Creating timeseries of dates
        dates = pd.date_range(t0, t1, freq="d")

        for index, row in stations.iterrows():
            # Similar to previous code, but only precipitation sum is stored

            a = df.tp.sel(
                latitude=row.Lat, longitude=row.Lon, method="nearest"
            ).to_dataset()

            arr = a.tp.values
            arr = arr.reshape(-1, 24)
            sum_ = np.sum(arr, 1)

            sdf = pd.DataFrame([dates, sum_]).T
            sdf.columns = ["time", "tp"]
            sdf.to_csv(saveloc + row.StationID + "_tp.csv", index=False)


def elevation_data(stations):
    # Elevation of the ERA5-Land gridcell is found through the geopotential
    # Elevation from SNOTEL is given in feet and needs to be converted to meter

    elev_xar = xr.open_dataset(
        "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc", engine="netcdf4"
    )
    # Printing xarray metadata
    for v in elev_xar:
        print(
            "{}, {}, {}".format(
                v, elev_xar[v].attrs["long_name"], elev_xar[v].attrs["units"]
            )
        )

    df = pd.DataFrame(columns=["StationID", "Elev", "Era5_elev"])
    for index, row in stations.iterrows():
        # Finding temperature using coordinates as index in xarray-df
        temp_df = elev_xar.z.sel(
            latitude=row.Lat, longitude=row.Lon + 360, method="nearest"
        ).to_dataset()

        # Converting to meters

        df.loc[len(df)] = [
            row.StationID,
            temp_df.z.values[0] / 9.80665,
            row.Elev * 0.3048,
        ]
        df.to_csv("USA_Elevations.csv", index=False)