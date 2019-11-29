#!/usr/bin/python3

import pandas as pd
import numpy as np
from datetime import datetime
from ReadOBS import ReadOBS_new
from ReadSIM import ReadSIM_new, ReadSIM_ECMWF,ReadSIM_ECMWF_F
import matplotlib.pyplot as plt
import statsmodels.api as sm


def FrameReadin():
    #福建石塘
    obs = ReadOBS_new(filename='./data/shitangObs_2019-UTC0.csv')
    obswind = obs['spd70'].dropna(axis=0,how='any')
    # convert type object to float64
    obswind = pd.to_numeric(obswind)

    simEC = ReadSIM_ECMWF_F(filename='./data_forecast_new/wind6houly201901-201903.csv')

    obswinds = obswind.reindex(simEC.index).dropna()
    simulations = simEC.reindex(obswind.index).dropna()

    df_all = pd.concat([obswinds, simulations], axis = 1).dropna()
    # # print(df_all)
    # features_all = list(df_all)
    # #print(df_all.info())
    # spdObs = df_all['spd70'].dropna(axis = 0)
    print("df_all:")
    print(df_all)
    print("df_all2:")
    df_all = df_all * 1.39
    df_all['spd70'] = df_all['spd70'] / 1.39
    print(df_all)
    return df_all

df_all = FrameReadin()

def plotobs(df_all):
    plotcols = ['spd70','WindSpeedVar1','WindSpeedVar28']
    df_plot = df_all[plotcols]

    # print(df_plot)
    df_plot.columns = ['spd70','WindSpeedVar1','WindSpeedVar28']
    data = df_plot

    # plot
    data.plot()
    #设置横纵坐标的名称以及对应字体格式
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 10,
    }
    plt.xlabel('time',font2)
    plt.ylabel('wind speed (m/s) ',font2)
    plt.legend(loc=1, prop={'size': 10})

    plt.savefig("./plot/ECMWF20190101/data.png")
    plt.clf()

plotobs(df_all)

# time shift 纠正时间漂移-------------------
def shitftime():
    def derive_ntime_feature(df,feature,N):
        rows = df.shape[0]
        ntime_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N,rows)]
        col_name = "{}_{}".format(feature,N)
        df[col_name] = ntime_prior_measurements

    for feature in features_all:
        if feature != 'date':
            for N in range(1,4):
                derive_ntime_feature(df_all,feature,N)


    to_remove = ['spd70_1', 'spd70_2', 'spd70_3']
    to_save = [col for col in df_all.columns if col not in to_remove]
    df_all = df_all[to_save]
# time shift 纠正时间漂移,样本量已经很多了，不用增加新的-------------------
# shitftime()

df_all = df_all.dropna(axis = 0)
# print(df_all)
# print(df_all.info())
# print(df_all.columns)
#从下式中选择相关性
#select features for our model
corr = df_all.corr()[['spd70']].sort_values('spd70')
print(corr)
# # choose predictors end corr > 0.5
predictors = ['WindSpeedVar1','WindSpeedVar28','WindSpeedVar12','WindSpeedVar10']

df2 = df_all[['spd70'] + predictors]
print("df22222222222222")
print(df2)

def corrfig():
    #-------------step 0 plot----------------------#
    #----------------------------------------------#
    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]

    # call subplots specifying the grid structure we desire and that 
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)

    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(2, 2)

    # use enumerate to loop over the arr 2D array of rows and columns
    # and create scatter plots of each meantempm vs each feature
    for row, col_arr in enumerate(arr):
        for col, feature in enumerate(col_arr):
            axes[row, col].scatter(df2[feature], df2['spd70'])
            if col == 0:
                # axes[row, col].set(xlabel=feature, ylabel='meantempm')
                axes[row, col].set(xlabel=feature, ylabel='speed70m')
            else:
                axes[row, col].set(xlabel=feature)
    #plt.show()
    plt.savefig("./plot/ECMWF20190101/corr.png")
    #-------------step 0 plot end----------------------#
corrfig()

#Using Step-wise Regression to Build a Robust Model
X = df2[predictors]
y = df2['spd70']

X = sm.add_constant(X)
#print(X.iloc[:5, :5])

alpha = 0.05
model = sm.OLS(y, X).fit()

# print(model.summary())

# 选择 p > 0.05，说明独立性不好的，删掉。

# 用回归模型做预报
from sklearn.model_selection import train_test_split

#移除常数项，因为sk-lean库不像statsmodels，会自动给我们添加一项
# first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
X = X.drop('const', axis=1)
print("XXXXXXXXXXX")
print(X)

X_train, _X_test, y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=12)

from sklearn.linear_model import LinearRegression
# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

X_test = df2[predictors] 
print(X_test) # no 1417 enough, all is 481

y_test = df2['spd70']
# make a prediction set using the test set
# y_test = df2['spd70'][1417:]
# make a prediction set using the test set
prediction = regressor.predict(X_test)

# ?? for what
# res=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index),only_fcst],axis=1).dropna()
# res=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index)],axis=1).dropna()
# print(res)

# print(df2)
prediction_origin_1 = df2['WindSpeedVar10']
prediction_origin_2 = df2['WindSpeedVar28']
prediction_origin_3 = df2['WindSpeedVar1']
prediction_origin_4 = df2['WindSpeedVar12']

#------------------
#y_test : 真实观测值,过后补全的; 
#prediction : 使用预报因子预报出来点结果
# origin 原来的3月份预报
#1
result1=pd.concat([y_test,pd.DataFrame({'end_var':prediction_origin_1},index=y_test.index)],axis=1).dropna()
result1.to_csv("./data/ECMWF20190101/output_obs_origin_1.csv",float_format="%.2f")
#2
result2=pd.concat([y_test,pd.DataFrame({'end_var':prediction_origin_2},index=y_test.index)],axis=1).dropna()
result2.to_csv("./data/ECMWF20190101/output_obs_origin_2.csv",float_format="%.2f")
#3
result3=pd.concat([y_test,pd.DataFrame({'end_var':prediction_origin_3},index=y_test.index)],axis=1).dropna()
result3.to_csv("./data/ECMWF20190101/output_obs_origin_3.csv",float_format="%.2f")
#4
result4=pd.concat([y_test,pd.DataFrame({'end_var':prediction_origin_4},index=y_test.index)],axis=1).dropna()
result4.to_csv("./data/ECMWF20190101/output_obs_origin_4.csv",float_format="%.2f")
# # no origin 订正后的3月份预报
result=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index)],axis=1).dropna()
result.to_csv("./data/ECMWF20190101/output_obs_prediction.csv",float_format="%.2f")

# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f m/s " % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f m/s " % median_absolute_error(y_test, prediction))
