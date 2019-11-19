# from pandas import Series
# from pandas import DataFrame
# from pandas import Grouper as TimeGrouper
from scipy.interpolate import pchip
from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# inputfile = './data/obs_prediction_3month.csv'
inputfile = './data/output_obs_prediction_3month_origin_one.csv'
def readindata(inputfile,names):
    with open(inputfile) as f:
        data = pd.read_csv(f,skiprows = 1, names=names)
        # data = pd.read_csv(f,skiprows = 1, names=names,index_col='time')
        return data

def datapandas():
    inputfile1 = './data/output_obs_prediction_3month_origin_one.csv'
    data1 = readindata(inputfile1,('time','spd70','before'))
    inputfile2 = './data/obs_prediction_3month.csv'
    data2 = readindata(inputfile2,('time','spd70','after'))
    # print(data1['time'].head())
    data1['time_parsed'] = pd.to_datetime(data1['time'],format="%Y-%m-%d %H:%M:%S")
    # print(data1['time_parsed'].head())
    data1['before'] = data1['before'] - 1.2
    # todo : if spd < 0; set spd = 0;
    # print(type(data1))
    data_origin = pd.merge(data1,data2)
    del data_origin['time']
    data = data_origin.set_index("time_parsed")
    data=data[data.before>0] # above 0
    return data

def plotfigTimeSeriesThreetoOne(data):
    plt = data.plot(lw=2,
                    title="wind predict before and after MOS ")
    plt.set_xlabel("time series")
    plt.set_ylabel("wind speed (m/s) ")

    fig = plt.get_figure()
    fig.savefig("./plot/TimeSeriesThreetoOne.png")
    # plt.clf()

def plotTimeGrouper(data):
    series  =  Series.from_csv('./data/output_obs_prediction_3month_origin_one.csv',  header=0)
    groups  =  series.groupby(TimeGrouper('A'))
    print(groups)
    years  =  DataFrame()

# output predict timeseries to csv.
# pchip 6hourly
def dataoutput(data): 
    print("dataoutput running")
    print(data['before'].max())
    print(data['before'].min())
    print(data['after'].max())
    print(data['after'].min())

    def pchip_prepare(data):
        data['timeindex'] = data.index # add a new column into data
        time_index_origin = data['timeindex']  # timeindex column
        # print(time_index_origin)
        time_index = pd.to_datetime(time_index_origin)  #convert type to datetime
        windpredict1h = data['after']
        return time_index, windpredict1h
    # train pchip obj
    def pchip6h(time_index, windpredict1h):
        pchip_obj_after1h = pchip(time_index, windpredict1h)
        # convert to 6houly time_index
        time_index_2 = pd.date_range(time_index.iloc[0], time_index.iloc[-1], freq = "360min")
        # test & predict 
        windpredict6h = pchip_obj_after1h(time_index_2) # * random.uniform(0.9,1.1) - 2
        
        time_index_2 = pd.Series(time_index_2)
        windpredict6h = pd.Series(windpredict6h)
        
        framelist = [time_index_2, windpredict6h]
        data6h = pd.concat(framelist, axis = 1)
        #print(data6h.iloc[:,0])
        return data6h
    
    def pchip6h_1h(time_index, windpredict1h):
        # got data6h data
        data6h = pchip6h(time_index, windpredict1h)
        # convert 6hourly to hourly
        # prepare data
        time_index_6h = data6h.iloc[:,0]
        winddata_6h = data6h.iloc[:,1]
        pchip_obj_6h = pchip(time_index_6h, winddata_6h)
        # convert to 6houly time_index
        time_index_1h = pd.date_range(time_index_6h.iloc[0], time_index_6h.iloc[-1], freq = "60min")
        # test & predict 
        windpredict1h = pchip_obj_6h(time_index_1h)
        # windpredict1h = pchip_obj_after1h(time_index_1h)
        time_index_1h = pd.Series(time_index_1h)
        windpredict1h = pd.Series(windpredict1h)
        framelist = [time_index_1h, windpredict1h]
        data1h = pd.concat(framelist, axis = 1)
        return data1h
    # output csv
    def pchip6h_output6h(data6h):
        data6h.to_csv('./data/timeseries6hourly.csv', index = False, float_format="%.2f")
    def pchip6h_output1h(data1h):
        data1h.to_csv('./data/timeseries1hourly.csv', index = False, float_format="%.2f")
   
    time_index, windpredict1h = pchip_prepare(data)
    data6h = pchip6h(time_index, windpredict1h)
    data1h = pchip6h_1h(time_index, windpredict1h)
    pchip6h_output6h(data6h)
    pchip6h_output1h(data1h)

def main():
    data=datapandas()
    plotfigTimeSeriesThreetoOne(data)
    dataoutput(data)
    # plotTimeGrouper(data) # todo

if __name__ == '__main__':
    main()