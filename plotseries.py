import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    data_origin = pd.merge(data1,data2)
    del data_origin['time']
    data = data_origin.set_index("time_parsed")
    print(data)
    
    # data = data_origin.set_index(time_parsed)
    data.to_csv('./data/timeseries.csv',float_format="%.2f")
    return data

def plotfig(data):
    plt = data.plot(lw=2,
                    title="wind predict before and after MOS ")
    plt.set_xlabel("time series")
    plt.set_ylabel("wind speed (m/s) ")

    fig = plt.get_figure()
    fig.savefig("./plot/TimeSeriesThreetoOne.png")
    # plt.clf()

def main():
    data=datapandas()
    plotfig(data)

if __name__ == '__main__':
    main()