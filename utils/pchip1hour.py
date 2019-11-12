#!/usr/bin/python3

import pandas as pd
import numpy as np
from datetime import datetime
# from ReadOBS import ReadOBS_new
# from ReadSIM import ReadSIM_new
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import pchip


def ReadOBS(filename=None):
    names = ['time','dir10','spd10','dir50','spd50','dir100','spd100','dir150','spd150']
    fhand = pd.read_csv(filename, skiprows = 1, 
                        names = names, 
                        usecols =[0,1,2,6,7,10,11,14,15])
    timeinfo = fhand['time']
    timeinfo_2 = timeinfo.replace('_', ' ', regex=True)
    timeinfo_panda = pd.to_datetime(timeinfo_2)
    dir10 = fhand['dir10']
    spd10 = fhand['spd10']
    dir50 = fhand['dir50']
    spd50 = fhand['spd50']
    dir100 = fhand['dir100']
    spd100 = fhand['spd100']
    dir150 = fhand['dir150']
    spd150 = fhand['spd150']
    frames = [timeinfo_panda, dir10, spd10, dir50, spd50, dir100, spd100, dir150, spd150]
    result0 = pd.concat(frames, axis = 1)
    result = result0.set_index('time')

    #intepolate to 1 hour
    time2_index=pd.date_range(timeinfo_panda.iloc[0], timeinfo_panda.iloc[-1], freq="60min")
    pchip_obj_spd10 = pchip(timeinfo_panda, spd10)
    pchip_obj_dir10 = pchip(timeinfo_panda, dir10)
    pchip_obj_spd50 = pchip(timeinfo_panda, spd50)
    pchip_obj_dir50 = pchip(timeinfo_panda, dir50)
    pchip_obj_spd100 = pchip(timeinfo_panda, spd100)
    pchip_obj_dir100 = pchip(timeinfo_panda, dir100)
    pchip_obj_spd150 = pchip(timeinfo_panda, spd150)
    pchip_obj_dir150 = pchip(timeinfo_panda, dir150)
    spd10_after = pchip_obj_spd10(time2_index)
    dir10_after = pchip_obj_dir10(time2_index)
    spd50_after = pchip_obj_spd50(time2_index)
    dir50_after = pchip_obj_dir50(time2_index)
    spd100_after = pchip_obj_spd100(time2_index)
    dir100_after = pchip_obj_dir100(time2_index)
    spd150_after = pchip_obj_spd150(time2_index)
    dir150_after = pchip_obj_dir150(time2_index)
    
    # convert columns to pandas series
    time2_index=pd.Series(time2_index)
    spd10_after = pd.Series(spd10_after)
    dir10_after = pd.Series(dir10_after)
    spd50_after = pd.Series(spd50_after)
    dir50_after = pd.Series(dir50_after)
    spd100_after = pd.Series(spd100_after)
    dir100_after = pd.Series(dir100_after)
    spd150_after = pd.Series(spd150_after)
    dir150_after = pd.Series(dir150_after)
    #out content:
    framelist=[time2_index, dir10_after, spd10_after, dir50_after, spd50_after, dir100_after, spd100_after, dir150_after, spd150_after]
    reanalfile2 = pd.concat(framelist, axis=1)
    Outhead = ["#   TimeInfo"," WindDirection10"," WindSpeed10"," WindDirection50"," WindSpeed50"," WindDirection100"," WindSpeed100"," WindDirection150"," WindSpeed150"]
    # print reanalfile

    filenameout = fileoutpath + outname
    reanalfile2.to_csv(filenameout, index = False, header = Outhead, encoding='utf-8') 
    return spd10_after

    
if __name__ == "__main__":
    # parameters: inname
    #EC
    # inname = "shitangSim_ECMWF_UTC0.csv"
    #GFS
    inname = "shitangSim_GFS_UTC0.csv"

    fileoutpath = "../data/"
    outname = "Hourly" + inname
    file = ReadOBS(filename = "../data/" + inname)
    print(file)