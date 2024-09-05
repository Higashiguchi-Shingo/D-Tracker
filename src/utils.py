import pandas as pd
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from statsmodels.tsa.seasonal import STL



#---------------------------------------------------#
# search(): 
#   dataframeを条件を指定して絞り込みするメソッド
#
# df2tts():
#   dfから(date, query, geo)のndarray配列に変換するメソッド
#   引数valueには"volume"を指定する
#---------------------------------------------------#
def date_search(df, date):
    ret = df[df["date"]==date]
    if len(ret)!=0:
        return ret
    else:
        sys.exit("Date:" + date + " does not exists.")

def query_search(df, query):
    ret = df[df["query"]==query]
    if len(ret)!=0:
        return ret
    else:
        sys.exit("Keyword:" + query + " does not exists.")

def country_search(df, country):
    ret = df[df["geo"]==country]
    if len(ret)!=0:
        return ret
    else:
        sys.exit("Country:" + country + " does not exists.")

def search(df, date=None, query=None, country=None):
    if date and query and country:
        return country_search(query_search(date_search(df, date), query), country)
    elif date and query and (not country):
        return query_search(date_search(df, date), query)
    elif date and (not query) and country:
        return country_search(date_search(df, date), country)
    elif (not date) and query and country:
        return country_search(query_search(df, query), country)
    elif (not date) and (not query) and country:
        return country_search(df, country)
    elif (not date) and query and (not country):
        return query_search(df, query)
    elif date and (not query) and (not country):
        return date_search(df, date)
    elif (not date) and (not query) and (not country):
        return df
    

#---------------------------------------------------#
# pathを指定してtensorを作成する関数
#---------------------------------------------------#
def load_tensor(path, time_key, facets, values=None, sampling_rate="D", start_date=None, end_date=None, scale="full"):
    df = pd.read_csv(path)
    tensor = df2tts(df, time_key=time_key, facets=facets, values=values, start_date=start_date, end_date=end_date)

    for key in facets:
        print(sorted(list(set(df[key]))))

    if scale=="full":
        tensor =  minmax_scale(tensor.reshape((-1, 1))).reshape(tensor.shape)
    elif scale=="each":
        tensor = min_max_scale_tensor(tensor)

    return tensor

def df2tts(df, time_key, facets, values=None, sampling_rate="D", start_date=None, end_date=None):
    """ Convert a DataFrame (list) to tensor time series

        df (pandas.DataFrame):
            A list of discrete events
        time_key (str):
            A column name of timestamps
        facets (list):
            A list of column names to make tensor timeseries
        values (str):
            A column name of target values (optional)
        sampling_rate (str):
            A frequancy for resampling, e.g., "7D", "12H", "H"
    """
    df[time_key] = pd.to_datetime(df[time_key])
    if start_date is not None: df = df[lambda x: x[time_key] >= pd.to_datetime(start_date)]
    if end_date is not None: df = df[lambda x: x[time_key] <= pd.to_datetime(end_date)]
    tmp = df.copy(deep=True)
    shape = tmp[facets].nunique().tolist()
    if values == None: values = 'count'; tmp[values] = 1
    tmp[time_key] = tmp[time_key].round(sampling_rate)
    print("Tensor:")
    print(tmp.nunique()[[time_key] + facets])

    grouped = tmp.groupby([time_key] + facets).sum()[[values]]
    grouped = grouped.unstack(fill_value=0).stack()
    grouped = grouped.pivot_table(index=time_key, columns=facets, values=values, fill_value=0)

    tts = grouped.values
    tts = np.reshape(tts, (-1, *shape))
    return tts

def select_country(df, geo_list):
    df_geo = pd.DataFrame(columns=["date","volume","query","geo"])
    for geo in geo_list:
        tmp = country_search(df, geo)
        df_geo = pd.concat([df_geo, tmp])
    df_geo["date"] = pd.to_datetime(df_geo["date"])
    return df_geo

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def ST_decomp(tensor, period):
    trend = np.zeros(shape=tensor.shape)
    seasonal = np.zeros(shape=tensor.shape)
    resid = np.zeros(shape=tensor.shape)

    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[2]):
            stl = STL(tensor[:,i,j], robust=True, period=period)
            stl_series = stl.fit()
            trend[:,i,j] = stl_series.trend
            seasonal[:,i,j] = stl_series.seasonal
            resid[:,i,j] = stl_series.resid
    
    return trend, seasonal, resid

def get_dim_list(path, mode="geo"):
    df = pd.read_csv(path)
    return sorted(list(set(df[mode])))

#----------------------------------------------------#
# 最大値1, 最小値0に正規化する関数
# min_max_scale_tensor()
# 入力:
#   data : 正規化前テンソル(time, keyword, location)
# 出力:
#   正規化したテンソル (time, keyword, location)
#----------------------------------------------------#
def min_max_scale_np(array):
    min = array.min()
    max = array.max()
    array = (array - min) / (max - min)
    return array

def min_max_scale_tensor(data):
    query_size = data.shape[1]
    geo_size = data.shape[2]
    ret = np.zeros(shape=data.shape)
    for i in range(query_size):
        for j in range(geo_size):
            ret[:,i,j] = min_max_scale_np(data[:,i,j])
    return ret
