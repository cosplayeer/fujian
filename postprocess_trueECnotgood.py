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
# inputfile = './data/output_obs_prediction_3month_origin_one.csv'
def readindata(inputfile,names):
    with open(inputfile) as f:
        data = pd.read_csv(f,skiprows = 1, names=names)
        # data = pd.read_csv(f,skiprows = 1, names=names,index_col='time')
        return data

def datapandas():
    # mainECandGFS_falseallEC1417month456.py
    #{
    # inputfile1 = './data/month456/output_obs_prediction_456month_origin_1.csv'
    # inputfile_after = './data/month456/output_obs_prediction_456month.csv'
    #}
    #mainECMWF_trueECnotgood.py
    #{
    inputfile1 = './data/true/output_obs_origin_1.csv'
    inputfile2 = './data/true/output_obs_origin_2.csv'
    inputfile3 = './data/true/output_obs_origin_3.csv'
    inputfile4 = './data/true/output_obs_origin_4.csv'
    inputfile_after = './data/true/output_obs_prediction.csv'
    #}
     # origin 原来的3月份预报
    data1 = readindata(inputfile1,('time','spd70','before')) # don't set to before1
    data2 = readindata(inputfile2,('time','spd70','before2'))
    data3 = readindata(inputfile3,('time','spd70','before3'))
    data4 = readindata(inputfile4,('time','spd70','before4'))
    # no origin 订正的3月份预报
    data_after = readindata(inputfile_after,('time','spd70','after'))
    # print(data1['time'].head())
    data1['time_parsed'] = pd.to_datetime(data1['time'],format="%Y-%m-%d %H:%M:%S")
    data2['time_parsed'] = pd.to_datetime(data2['time'],format="%Y-%m-%d %H:%M:%S")
    data3['time_parsed'] = pd.to_datetime(data3['time'],format="%Y-%m-%d %H:%M:%S")
    data4['time_parsed'] = pd.to_datetime(data4['time'],format="%Y-%m-%d %H:%M:%S")
    data_after['time_parsed'] = pd.to_datetime(data_after['time'],format="%Y-%m-%d %H:%M:%S")
    #data1['before'] = data1['before'] #- 1.2 # origin data as slow as true predictors
    # print(data1['time_parsed'].head())
    # if spd < 0; set spd = 0;
    # print(type(data1))
    # data build {
    data_origin = pd.merge(data1,data_after)
    del data_origin['time']
    data = data_origin.set_index("time_parsed")
    # data=data[data.before>0] # above 0
    #}
    # data_four more build {
    data_more_1 = pd.merge(data1, data2)
    data_more_2 = pd.merge(data_more_1, data3)
    data_more_3 = pd.merge(data_more_2, data4)
    data_more_origin = pd.merge(data_more_3, data_after)
    # data_more_origin = pd.concat([data1, data2, data3, data4, data_after], axis = 1)
    del data_more_origin['time']
    data_more = data_more_origin.set_index("time_parsed")
    data_two = data_more
    data_four = data_more
    del data_four['spd70']
    del data_four['after']
    # del data_two['before']
    # del data_two['before2']
    # del data_two['before3']
    # del data_two['before4']
    print(data_four)
    del data_after['time']
    data_two = data_after.set_index("time_parsed")
    print(data_two)
    #}
    # print("type data")
    # print(type(data))
    # print("type data_two")
    # print(type(data_two))
    return data, data_two, data_four 

def plotfigTimeSeriesThreetoOne(data):
    plt = data.plot(lw=2,
                    title="wind predict before and after MOS ")
    plt.set_xlabel("time series")
    plt.set_ylabel("wind speed (m/s) ")

    fig = plt.get_figure()
    fig.savefig("./plot/true/TimeSeriesThreetoOne.png")
    fig.clf()

def plotfigTimeSeriesSixtoOne(data_two, data_four):
    ax = data_two.plot(lw=2,
                    kind = 'line',
                    alpha=1,
                    title="wind predict before and after MOS ")
    data_four.plot(lw=2,
                kind = 'line',
                alpha=0.15,
                ax = ax)
    ax.set_xlabel("time series")
    ax.set_ylabel("wind speed (m/s) ")

    fig = ax.get_figure()
    fig.savefig("./plot/true/TimeSeriesSixtoOne.png")
    fig.clf()

def plotTimeGrouper(data):
    series  =  Series.from_csv('./data/output_obs_prediction_3month_origin_one.csv',  header=0)
    groups  =  series.groupby(TimeGrouper('A'))
    print(groups)
    years  =  DataFrame()

def plotfigbox(data_two):
    s = data_two['spd70']
    # print(s)
    ax1 = s.hist()
    fig = ax1.get_figure()
    fig.savefig("./plot/true/hist.png")
    fig.clf()
    
    s2 = data_two['after']
    ax1 = s2.hist()
    fig = ax1.get_figure()
    fig.savefig("./plot/true/hist_after.png")
    fig.clf()
# output predict timeseries to csv.
# pchip 6hourly
def dataoutput(data): 
    print("dataoutput running:")
    # print(data['before'].max())
    # print(data['before'].min())
    print("average of obs:")
    print(data['spd70'].mean())
    print("average before mos:")
    print(data['before'].mean())
    # print(data['after'].max())
    # print(data['after'].min())
    print("average after mos:")
    print(data['after'].mean())

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
        # test & predict, dyp
        windpredict6h = pchip_obj_after1h(time_index_2)  #* random.uniform(0.9,1.1) - 2 # if < 0 : set to 0.
        # windpredict6h = pchip_obj_after1h(time_index_2) * random.uniform(0.9,1.1) *0.8 # 相关性变差，影响最大最小

        print("MMMiiiiiiiiiiiiiii of windpredict6h:")
        print(windpredict6h.min())
        
        time_index_2 = pd.Series(time_index_2)
        windpredict6h = pd.Series(windpredict6h)
        
        framelist = [time_index_2, windpredict6h]
        data6h = pd.concat(framelist, axis = 1)
        #print(data6h.iloc[:,0])
        print("average of data6h after mos:")
        print("should be the same as above.")
        print(data6h.mean())
        # data6h = data6h[data6h.windpredict6h > 0]
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
        time_index_1h = pd.date_range(time_index_6h.iloc[0], time_index_6h.iloc[-1], freq = "15min")
        # test & predict 
        windpredict1h = pchip_obj_6h(time_index_1h)

        print("MMMiiiiiiiiiiiiiii of windpredict15m:")
        print(windpredict1h.min())
        # windpredict1h = pchip_obj_after1h(time_index_1h)
        time_index_1h = pd.Series(time_index_1h)
        windpredict1h = pd.Series(windpredict1h)
        framelist = [time_index_1h, windpredict1h]
        data1h = pd.concat(framelist, axis = 1)
        return data1h
    # output csv
    def pchip6h_output6h(data6h):
        data6h.to_csv('./data/true/timeseries6hourly.csv', index = False, float_format="%.2f")
    def pchip6h_output1h(data1h):
        data1h.to_csv('./data/true/timeseries1hourly.csv', index = False, float_format="%.2f")
   
    time_index, windpredict1h = pchip_prepare(data)
    data6h = pchip6h(time_index, windpredict1h)
    data1h = pchip6h_1h(time_index, windpredict1h)
    pchip6h_output6h(data6h)
    pchip6h_output1h(data1h)

def main():
    data, data_two, data_four =datapandas()
    plotfigTimeSeriesThreetoOne(data)
    plotfigTimeSeriesSixtoOne(data_two, data_four)
    plotfigbox(data_two)

    # output csv
    dataoutput(data)
    # plotTimeGrouper(data) # todo

if __name__ == '__main__':
    main()