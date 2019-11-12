import numpy as np
import pandas as pd
from datetime import datetime

def ReadSIMGFS(filename=None):
    namelists = ['TimeInfo',     'GFSWindDirection10',     'GFSWindSpeed10',     'GFSWindDirection50',   
    'GFSWindSpeed50',   'GFSWindDirection100',   'GFSWindSpeed100',  'GFSWindDirection150',  'GFSWindSpeed150']
    fhand = pd.read_csv(filename, 
            skiprows = 1, names = namelists, usecols =[0,1,2,3,4,5,6,7,8])
    fhand = fhand.replace('NAN',np.nan)
    fhand = fhand.dropna()
    timeinfo = fhand['TimeInfo']
    # stringtime = timeinfo.str.replace('_', ' ')
    stringtime = timeinfo
    timeinfo_panda = pd.to_datetime(stringtime)
    dir10 = fhand['GFSWindDirection10']
    v10 = fhand['GFSWindSpeed10']
    dir50 = fhand['GFSWindDirection50']
    v50 = fhand['GFSWindSpeed50']
    dir100 = fhand['GFSWindDirection100']
    v100 = fhand['GFSWindSpeed100']
    dir150 = fhand['GFSWindDirection150']
    v150 = fhand['GFSWindSpeed150']
    
    # Time Shift
    features_all = [timeinfo_panda,dir10,
            v10,dir50,v50,dir100,
            v100,dir150,v150]
    result0 = pd.concat(features_all, axis = 1)
    #print(result)
    result = result0.set_index('TimeInfo')
    return result

def ReadSIMEC(filename=None):
    namelists = ['TimeInfo',     'ECWindDirection10',     'ECWindSpeed10',     'ECWindDirection50',   
    'ECWindSpeed50',   'ECWindDirection100',   'ECWindSpeed100',  'ECWindDirection150',  'ECWindSpeed150']
    fhand = pd.read_csv(filename, 
            skiprows = 1, names = namelists, usecols =[0,1,2,3,4,5,6,7,8])
    fhand = fhand.replace('NAN',np.nan)
    fhand = fhand.dropna()
    timeinfo = fhand['TimeInfo']
    # stringtime = timeinfo.str.replace('_', ' ')
    stringtime = timeinfo
    timeinfo_panda = pd.to_datetime(stringtime)
    dir10 = fhand['ECWindDirection10']
    v10 = fhand['ECWindSpeed10']
    dir50 = fhand['ECWindDirection50']
    v50 = fhand['ECWindSpeed50']
    dir100 = fhand['ECWindDirection100']
    v100 = fhand['ECWindSpeed100']
    dir150 = fhand['ECWindDirection150']
    v150 = fhand['ECWindSpeed150']
    
    # Time Shift
    features_all = [timeinfo_panda,dir10,
            v10,dir50,v50,dir100,
            v100,dir150,v150]
    result0 = pd.concat(features_all, axis = 1)
    #print(result)
    result = result0.set_index('TimeInfo')
    return result

def ReadSIM_new(filename=None):
    namelists = ['TimeInfo',     'WindDirection10',     'WindSpeed10',     'WindDirection50',   
    'WindSpeed50',   'WindDirection100',   'WindSpeed100',  'WindDirection150',  'WindSpeed150']
    fhand = pd.read_csv(filename, 
            skiprows = 1, names = namelists, usecols =[0,1,2,3,4,5,6,7,8])
    fhand = fhand.replace('NAN',np.nan)
    fhand = fhand.dropna()
    timeinfo = fhand['TimeInfo']
    # stringtime = timeinfo.str.replace('_', ' ')
    stringtime = timeinfo
    timeinfo_panda = pd.to_datetime(stringtime)
    dir10 = fhand['WindDirection10']
    v10 = fhand['WindSpeed10']
    dir50 = fhand['WindDirection50']
    v50 = fhand['WindSpeed50']
    dir100 = fhand['WindDirection100']
    v100 = fhand['WindSpeed100']
    dir150 = fhand['WindDirection150']
    v150 = fhand['WindSpeed150']
    
    # Time Shift
    features_all = [timeinfo_panda,dir10,
            v10,dir50,v50,dir100,
            v100,dir150,v150]
    result0 = pd.concat(features_all, axis = 1)
    #print(result)
    df = result0.set_index('TimeInfo')

    return df

def ReadSIM_ECMWF(filename=None):
#     namelists = ['TimeInfo', 'WindSpeedVar1', 'WindSpeedVar2', 'WindSpeedVar3', 'WindSpeedVar4', 'WindSpeedVar5', 'WindSpeedVar6', 'WindSpeedVar7', 'WindSpeedVar8','WindSpeedVar9','WindSpeedVar10',
#                               'WindSpeedVar11', 'WindSpeedVar12', 'WindSpeedVar13', 'WindSpeedVar14', 'WindSpeedVar15', 'WindSpeedVar16', 'WindSpeedVar17', 'WindSpeedVar18','WindSpeedVar19','WindSpeedVar20',
#                               'WindSpeedVar21', 'WindSpeedVar22', 'WindSpeedVar23', 'WindSpeedVar24', 'WindSpeedVar25', 'WindSpeedVar26', 'WindSpeedVar27', 'WindSpeedVar28','WindSpeedVar29','WindSpeedVar30',
#                               'WindSpeedVar31', 'WindSpeedVar32', 'WindSpeedVar33', 'WindSpeedVar34', 'WindSpeedVar35', 'WindSpeedVar36', 'WindSpeedVar37', 'WindSpeedVar38','WindSpeedVar39','WindSpeedVar40',
#                               'WindSpeedVar41', 'WindSpeedVar42', 'WindSpeedVar43', 'WindSpeedVar44', 'WindSpeedVar45', 'WindSpeedVar46', 'WindSpeedVar47', 'WindSpeedVar48','WindSpeedVar49','WindSpeedVar50',
#                               'WindSpeedVar51']
    namelists = ['TimeInfo', 'WindSpeedVar1', 'WindSpeedVar2', 'WindSpeedVar3', 'WindSpeedVar4', 'WindSpeedVar5', 'WindSpeedVar6', 'WindSpeedVar7', 'WindSpeedVar8','WindSpeedVar9','WindSpeedVar10',
                                'WindSpeedVar11', 'WindSpeedVar12', 'WindSpeedVar13', 'WindSpeedVar14', 'WindSpeedVar15', 'WindSpeedVar16', 'WindSpeedVar17', 'WindSpeedVar18','WindSpeedVar19','WindSpeedVar20',
                                'WindSpeedVar21', 'WindSpeedVar22', 'WindSpeedVar23', 'WindSpeedVar24', 'WindSpeedVar25', 'WindSpeedVar26', 'WindSpeedVar27', 'WindSpeedVar28','WindSpeedVar29','WindSpeedVar30',
                                'WindSpeedVar31', 'WindSpeedVar32'] #, 'WindSpeedVar33', 'WindSpeedVar34']
                                #, 'WindSpeedVar35']#, 'WindSpeedVar36', 'WindSpeedVar37', 'WindSpeedVar38','WindSpeedVar39','WindSpeedVar40']
    fhand = pd.read_csv(filename, skiprows = 1, names = namelists, usecols =[0,1,2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20,21,22,23,34,25,26,27,28,29,30,31,32]) 
    #,35,36,37,38,39,40])
    #     skiprows = 1, names = namelists, usecols =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,34,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51])
    print(fhand)
#     return 0 

    fhand = fhand.replace('NAN',np.nan)
    fhand = fhand.dropna()
    timeinfo = fhand['TimeInfo']

    stringtime = timeinfo
    timeinfo_panda = pd.to_datetime(stringtime)
    WindSpeedVar1 = fhand['WindSpeedVar1']
    WindSpeedVar2 = fhand['WindSpeedVar2']
    WindSpeedVar3 = fhand['WindSpeedVar3']
    WindSpeedVar4 = fhand['WindSpeedVar4']
    WindSpeedVar5 = fhand['WindSpeedVar5']
    WindSpeedVar6 = fhand['WindSpeedVar6']
    WindSpeedVar7 = fhand['WindSpeedVar7']
    WindSpeedVar8 = fhand['WindSpeedVar8']
    WindSpeedVar9 = fhand['WindSpeedVar9']
    WindSpeedVar10 = fhand['WindSpeedVar10']
    WindSpeedVar11 = fhand['WindSpeedVar11']
    WindSpeedVar12 = fhand['WindSpeedVar12']
    WindSpeedVar13 = fhand['WindSpeedVar13']
    WindSpeedVar14 = fhand['WindSpeedVar14']
    WindSpeedVar15 = fhand['WindSpeedVar15']
    WindSpeedVar16 = fhand['WindSpeedVar16']
    WindSpeedVar17 = fhand['WindSpeedVar17']
    WindSpeedVar18 = fhand['WindSpeedVar18']
    WindSpeedVar19 = fhand['WindSpeedVar19']
    WindSpeedVar20 = fhand['WindSpeedVar20']
    WindSpeedVar21 = fhand['WindSpeedVar21']
    WindSpeedVar22 = fhand['WindSpeedVar22']
    WindSpeedVar23 = fhand['WindSpeedVar23']
    WindSpeedVar24 = fhand['WindSpeedVar24']
    WindSpeedVar25 = fhand['WindSpeedVar25']
    WindSpeedVar26 = fhand['WindSpeedVar26']
    WindSpeedVar27 = fhand['WindSpeedVar27']
    WindSpeedVar28 = fhand['WindSpeedVar28']
    WindSpeedVar29 = fhand['WindSpeedVar29']
    WindSpeedVar30 = fhand['WindSpeedVar30']
    WindSpeedVar31 = fhand['WindSpeedVar31']
    WindSpeedVar32 = fhand['WindSpeedVar3']
#     WindSpeedVar33 = fhand['WindSpeedVar33']
#     WindSpeedVar34 = fhand['WindSpeedVar34']
#     WindSpeedVar35 = fhand['WindSpeedVar35']
#     WindSpeedVar36 = fhand['WindSpeedVar36']
#     WindSpeedVar37 = fhand['WindSpeedVar37']
#     WindSpeedVar38 = fhand['WindSpeedVar38']
#     WindSpeedVar39 = fhand['WindSpeedVar39']
#     WindSpeedVar40 = fhand['WindSpeedVar40']
#     WindSpeedVar41 = fhand['WindSpeedVar41']
#     WindSpeedVar42 = fhand['WindSpeedVar42']
#     WindSpeedVar43 = fhand['WindSpeedVar43']
#     WindSpeedVar44 = fhand['WindSpeedVar44']
#     WindSpeedVar45 = fhand['WindSpeedVar45']
#     WindSpeedVar46 = fhand['WindSpeedVar46']
#     WindSpeedVar47 = fhand['WindSpeedVar47']
#     WindSpeedVar48 = fhand['WindSpeedVar48']
#     WindSpeedVar49 = fhand['WindSpeedVar49']
#     WindSpeedVar50 = fhand['WindSpeedVar50']
#     WindSpeedVar51 = fhand['WindSpeedVar51']
    
#     # Time Shift
    features_all = [timeinfo_panda,
            WindSpeedVar1,WindSpeedVar2,WindSpeedVar3,WindSpeedVar4,WindSpeedVar5,WindSpeedVar6,WindSpeedVar7,WindSpeedVar8, WindSpeedVar9, WindSpeedVar10,
            WindSpeedVar11,WindSpeedVar12,WindSpeedVar13,WindSpeedVar14,WindSpeedVar15,WindSpeedVar16,WindSpeedVar17,WindSpeedVar18,WindSpeedVar19,WindSpeedVar20,
            WindSpeedVar21,WindSpeedVar22,WindSpeedVar23,WindSpeedVar24,WindSpeedVar25,WindSpeedVar26,WindSpeedVar27,WindSpeedVar28,WindSpeedVar29,WindSpeedVar30,
            WindSpeedVar31,WindSpeedVar32]
        #     ,WindSpeedVar33,WindSpeedVar34,WindSpeedVar35,WindSpeedVar36,WindSpeedVar37,WindSpeedVar38,WindSpeedVar39,WindSpeedVar40,
        #     WindSpeedVar41,WindSpeedVar42,WindSpeedVar43,WindSpeedVar44,WindSpeedVar45,WindSpeedVar46,WindSpeedVar47,WindSpeedVar48,WindSpeedVar49,WindSpeedVar50,
        #     WindSpeedVar51]
    result0 = pd.concat(features_all, axis = 1)
    #print(result)
    df = result0.set_index('TimeInfo')

    return df


#Test
if __name__ == "__main__":
#     df = ReadSIM_ECMWF(filename='./data/6HourlyshitangSim_ECMWF_UTC0.csv')
#     print(df)
    f = ReadSIM_ECMWF(filename='./data/6HourlyshitangSim_ECMWF_UTC0.csv')