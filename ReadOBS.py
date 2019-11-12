import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import Day, Hour


def ReadOBS(filename=None):
    #fhand = pd.read_csv('./data/testObs1.csv',index_col = 'time',usecols =[0,1,2])
    names = ['time', 'spd70','dir70']
    fhand = pd.read_csv(filename, skiprows = 1, 
                        names = names, usecols =[0,5,6])
    fhand = fhand.replace('NAN',np.nan)
    fhand = fhand.dropna()
    timeinfo = fhand['time']
    spd = fhand['spd70']
    dir = fhand['dir70']
    timeinfo_2 = timeinfo.replace('_', ' ', regex=True)
    timeinfo_panda = pd.to_datetime(timeinfo_2)
    timeinfo_pandaUTC0 = timeinfo_panda 
    # if date is UTC8, not UTC0
    # timeinfo_pandaUTC0 = timeinfo_panda - 8 * Hour()
    #timeinfo_pandaUTC0 = timeinfo_panda.shift(-8, freq='H')
    #time1 = [ datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in timeinfo ]
    
    frames = [timeinfo_pandaUTC0, spd]
    result0 = pd.concat(frames, axis = 1)
    result = result0.set_index('time')
    return result

def ReadOBS_new(filename=None):
    #fhand = pd.read_csv('./data/testObs1.csv',index_col = 'time',usecols =[0,1,2])
    names = ['time', 'spd70']
    fhand = pd.read_csv(filename, skiprows = 1, 
            names = names, usecols =[0,5])
    fhand = fhand.replace('NAN',np.nan)
    fhand = fhand.dropna()
    # fhand = pd.read_csv('./data_ec2017/M1525-Exported-timeleft.csv', skiprows = 2, 
    #         names = names, usecols =[0,1])
    timeinfo = fhand['time']
    spd = fhand['spd70']

    stringtime = timeinfo.str.replace('_', ' ')
    timeinfo_panda = pd.to_datetime(stringtime)
    # time shifted already, so commoned.
    #timeinfo_pandaUTC0 = timeinfo_panda - 8 * Hour()
    timeinfo_pandaUTC0 = timeinfo_panda 
    
    #time1 = [ datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in timeinfo ]
    #print(time1)
    frames = [timeinfo_pandaUTC0, spd]
    result0 = pd.concat(frames, axis = 1)
    result = result0.set_index('time').dropna()
    # x = result.isnull().sum().sum()
    # # print(result['2019-01-01 00:00:00'])
    # print("hi")
    # print(x)
    print(result)
    return result

if __name__ == "__main__":
    result = ReadOBS_new(filename="./data/shitangObs_2019-UTC0.csv")
    print(type(result))