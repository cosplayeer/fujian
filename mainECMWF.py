#!/usr/bin/python3

import pandas as pd
import numpy as np
from datetime import datetime
from ReadOBS import ReadOBS_new
from ReadSIM import ReadSIM_new, ReadSIM_ECMWF
import matplotlib.pyplot as plt
import statsmodels.api as sm
#福建石塘
obs = ReadOBS_new(filename='./data/shitangObs_2019-UTC0.csv')
obswind = obs['spd70'].dropna(axis=0,how='any')
# convert type object to float64
obswind = pd.to_numeric(obswind)

simEC = ReadSIM_ECMWF(filename='./data/6HourlyshitangSim_ECMWF_UTC0.csv')

obswinds = obswind.reindex(simEC.index).dropna()
simulations = simEC.reindex(obswind.index).dropna()

df_all = pd.concat([obswinds, simulations], axis = 1).dropna()
# print(df_all)
features_all = list(df_all)
#print(df_all.info())
spdObs = df_all['spd70'].dropna(axis = 0)
print(spdObs)

# plot
plt.plot(spdObs)
plt.savefig("./data/hello.png")
plt.clf()


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
print(df_all)
print(df_all.info())
print(df_all.columns)
#从下式中选择相关性
#select features for our model
corr = df_all.corr()[['spd70']].sort_values('spd70')
print(corr)
# # choose predictors end corr > 0.5
predictors = ['WindSpeedVar1','WindSpeedVar2','WindSpeedVar3','WindSpeedVar4','WindSpeedVar5',
            'WindSpeedVar6','WindSpeedVar7','WindSpeedVar8','WindSpeedVar9','WindSpeedVar10',
            'WindSpeedVar11','WindSpeedVar12','WindSpeedVar13','WindSpeedVar14','WindSpeedVar15',
            'WindSpeedVar16','WindSpeedVar17','WindSpeedVar18','WindSpeedVar19','WindSpeedVar20',
            'WindSpeedVar21','WindSpeedVar22','WindSpeedVar23','WindSpeedVar24','WindSpeedVar25',
            'WindSpeedVar26','WindSpeedVar27','WindSpeedVar28','WindSpeedVar29','WindSpeedVar30']
            # 'WindSpeedVar31','WindSpeedVar32']
            # ,'WindSpeedVar33','WindSpeedVar34''WindSpeedVar35',
            # 'WindSpeedVar36','WindSpeedVar37','WindSpeedVar38','WindSpeedVar39''WindSpeedVar40',
            # 'WindSpeedVar41','WindSpeedVar42','WindSpeedVar43','WindSpeedVar44''WindSpeedVar45',
            # 'WindSpeedVar46','WindSpeedVar47','WindSpeedVar48','WindSpeedVar49''WindSpeedVar50',
            # 'WindSpeedVar51']

df2 = df_all[['spd70'] + predictors]
print(df2)

def corrfig():
    #-------------step 0 plot----------------------#
    #----------------------------------------------#
    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [16, 22]

    # call subplots specifying the grid structure we desire and that 
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=5, ncols=6, sharey=True)

    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(5, 6)

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
    plt.savefig("./data/corr.png")
    #-------------step 0 plot end----------------------#
corrfig()

#Using Step-wise Regression to Build a Robust Model
X = df2[predictors]
y = df2['spd70']

X = sm.add_constant(X)
#print(X.iloc[:5, :5])

alpha = 0.05
model = sm.OLS(y, X).fit()

print(model.summary())

# 选择 p > 0.05，说明独立性不好的，删掉。

# 用回归模型做预报
from sklearn.model_selection import train_test_split

#移除常数项，因为sk-lean库不像statsmodels，会自动给我们添加一项
# first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
X = X.drop('const', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

from sklearn.linear_model import LinearRegression
# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# res=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index),only_fcst],axis=1).dropna()
res=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index)],axis=1).dropna()
print(res)

# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f m/s " % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f m/s " % median_absolute_error(y_test, prediction))
