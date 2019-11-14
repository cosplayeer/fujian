#!/usr/bin/python3

import matplotlib
import pandas as pd
import numpy as np
from datetime import datetime
from ReadOBS import ReadOBS_new
from ReadSIM import ReadSIMEC,ReadSIMGFS
import matplotlib.pyplot as plt
import statsmodels.api as sm

def FrameReadin():
    #福建石塘obs
    obs = ReadOBS_new(filename='./data/shitangObs_2019-UTC0.csv')
    obswind = obs['spd70'].dropna(axis=0,how='any')
    # convert type object to float64
    obswind = pd.to_numeric(obswind)
    # print(type(obswind))
    #福建石塘sim
    simGFS = ReadSIMGFS(filename='./data/HourlyshitangSim_GFS_UTC0.csv')
    simulationsA = simGFS.reindex(obswind.index).dropna()
    simEC = ReadSIMEC(filename='./data/HourlyshitangSim_ECMWF_UTC0.csv')
    simulationsB = simEC.reindex(obswind.index).dropna()

    df_all = pd.concat([obswind, simulationsA, simulationsB], axis = 1).dropna()
    return df_all
    print(df_all)

    print(df_all.info())

df_all = FrameReadin()

def plotobs():
    spdObs = df_all['spd70'].dropna(axis = 0)
    # ECWindSpeed100 = df_all['ECWindSpeed100'].dropna(axis = 0)
    plotcols = ['spd70','ECWindSpeed10','ECWindSpeed50','ECWindSpeed100','ECWindSpeed150']
    df_plot = df_all[plotcols]
    df_plot.columns = ['obs','ens1','ens2','ens3','ens4']
    # data = pd.DataFrame(df_plot, columns = list("ABCD"))
    data = df_plot
    
    print(data)
    # plot
    # plt.plot(data)
    data.plot()
    #设置横纵坐标的名称以及对应字体格式
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 10,
    }
    plt.xlabel('time',font2)
    plt.ylabel('wind speed (m/s) ',font2)
    plt.legend(loc=1, prop={'size': 10})

    plt.savefig("./plot/data.png")
    plt.clf()

plotobs()

def derive_ntime_feature(df,feature,N):
    rows = df.shape[0]
    ntime_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N,rows)]
    col_name = "{}_{}".format(feature,N)
    df[col_name] = ntime_prior_measurements

def derive_ntime_feature_df_all():
    features_all = list(df_all)
    for feature in features_all:
        if feature != 'date':
            for N in range(1,4):
                derive_ntime_feature(df_all,feature,N)

derive_ntime_feature_df_all()

to_remove = ['spd70_1', 'spd70_2', 'spd70_3']
to_save = [col for col in df_all.columns if col not in to_remove]
df_all = df_all[to_save]
df_all = df_all.dropna(axis = 0)
# print(df_all)
# print(df_all.info())
# print(df_all.columns)
def SelectFeatures():
    #从下式中选择相关性
    #select features for our model
    corr = df_all.corr()[['spd70']].sort_values('spd70')
    print(corr)
    # # choose predictors end corr > 0.5
    #M1525
    # predictors = ['speed','windpower','windpower_1','speed_1','windpower_2','speed_2','windpower_3','speed_3',]
    predictorsA = ['GFSWindSpeed10','GFSWindSpeed50','GFSWindSpeed100','GFSWindSpeed100_1','GFSWindSpeed100_2','GFSWindSpeed150','GFSWindSpeed150_1']
    predictorsB = ['ECWindSpeed10','ECWindSpeed50','ECWindSpeed100','ECWindSpeed100_1','ECWindSpeed100_2','ECWindSpeed150','ECWindSpeed150_1','ECWindSpeed150_2']
    predictors = predictorsA + predictorsB
    predictors2 = ['EC1WindSpeed','EC2WindSpeed','EC3WindSpeed','EC3WindSpeed_1','EC3WindSpeed_2',
                    'EC4WindSpeed','EC4WindSpeed_1','EC5WindSpeed','EC6WindSpeed','EC7WindSpeed',
                    'EC7WindSpeed_1','EC7WindSpeed_2','EC8WindSpeed','EC8WindSpeed_1','EC8WindSpeed_2']

    df2 = df_all[['spd70'] + predictors]
    df2.columns = ['spd70','EC1WindSpeed','EC2WindSpeed','EC3WindSpeed','EC3WindSpeed_1','EC3WindSpeed_2',
                    'EC4WindSpeed','EC4WindSpeed_1','EC5WindSpeed','EC6WindSpeed','EC7WindSpeed',
                    'EC7WindSpeed_1','EC7WindSpeed_2','EC8WindSpeed','EC8WindSpeed_1','EC8WindSpeed_2']
    print(df2)
    print(df2.info())
    print(df2.columns)
    return predictors2, df2

predictors, df2 = SelectFeatures()    
#-------------step 0 plot----------------------#
#----------------------------------------------#
# manually set the parameters of the figure to and appropriate size
def plotcorrpanel():
    plt.rcParams['figure.figsize'] = [16, 22]

    # call subplots specifying the grid structure we desire and that 
    # the y axes should be shared
    fig, axes = plt.subplots(nrows=5, ncols=3, sharey=True)

    # Since it would be nice to loop through the features in to build this plot
    # let us rearrange our data into a 2D array of 6 rows and 3 columns
    arr = np.array(predictors).reshape(5, 3)

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
    plt.savefig("plot/corrpanel.png")
    plt.clf()
    #-------------step 0 plot end----------------------#

plotcorrpanel()

def plotscatter():
    #设置输出的图片大小
    figsize = 11,9
    figure, ax = plt.subplots(figsize=figsize)

    data1 = df2[['ECWindSpeed10','spd70']]
    # data1 = df2[['ECWindSpeed10','spd70']]
    data1.columns = ['ens1','obs']
    pic1 = data1.plot.scatter(x='obs',y='ens1',color='DarkBlue', label="Ensemble 1")
    # pic1.legend('CCCC',fontsize='large') 
    # pic1.legend(loc=2, prop={'size': 20})

    data2 = df2[['ECWindSpeed100','spd70']]
    # data2 = df2[['ECWindSpeed100','spd70']]
    data2.columns = ['ens2','obs']
    data2.plot.scatter(x='obs',y='ens2',color='DarkGreen', label="Ensemble 2", ax=pic1)

    # data3 = df2[['GFSWindSpeed100','spd70']]
    # data3.columns = ['ens3','obs']
    # data3.plot.scatter(x='obs',y='ens3',color='Black', label="Class 3", ax=pic1)

    data4 = df2[['GFSWindSpeed10','spd70']]
    # data4 = df2[['GFSWindSpeed10','spd70']]
    data4.columns = ['ens4','obs']
    data4.plot.scatter(x='obs',y='ens4',color='DarkRed', label="Ensemble 3", ax=pic1)
    # plt.plot(data3['obs'],data3['ens3'])
    # plt.rcParams['figure.figsize'] = [22, 16]
    # plt.rcParams['figure.figsize'] = [6.4, 4.8]

    # #设置图例并且设置图例的字体及大小
    # font1 = {'family' : 'Times New Roman',
    # 'weight' : 'normal',
    # 'size'   : 23,
    # }
    # legend = plt.legend(handles=["A","B","C","D"],prop=font1)


    #设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    #设置横纵坐标的名称以及对应字体格式
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }
    plt.xlabel('obs wind at 70 meters',font2)
    plt.ylabel('part of forecast ensembles ',font2)
    plt.legend(loc=1, prop={'size': 30})

    # matplotlib.rc('xtick',labelsize=100)
    plt.savefig("./plot/scatter.png")
    plt.clf()

# plotscatter()

def plothot1():
    import seaborn as sns
    sns.set()

    uniform_data = df2
    # 改变颜色映射的值范围
    ax = sns.heatmap(uniform_data, cmap='YlGnBu')
    plt.savefig("plot/hotfig.png")
    plt.clf()

# plothot1()    

def plothot2():
    import seaborn as sns
    sns.set()
    corr = np.corrcoef(df2,rowvar=0)
    print(corr.dtype)
    print(np.size(corr))
    print(np.size(corr,0))
    print(np.size(corr,1))
    print("hhhhhhhhhhhhhhh")
    for i in range(16):
        for j in range(16):
            if corr[i,j] < 0.9999:
                corr[i,j] = corr[i,j] * 0.7
            else:
                corr[i,j] = corr[i,j] 
    print(corr)
    fig, ax = plt.subplots(1, 1)
    # cbar_ax = fig.add_axes([.905, .3, .05, .3])
    cbar_ax = fig.add_axes([.93, .22, .025, .55])
    # print(corr.info)
    uniform_data = corr
    # 改变颜色映射的值范围
    # cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    # sns.heatmap(uniform_data, ax=ax, cbar_ax = cbar_ax, square=True, linewidths = 0.05,vmax=1, cmap=cmap) # vmax=1, vmin=0,
    cmap="YlGnBu"
    sns.heatmap(uniform_data, ax=ax, cbar_ax = cbar_ax, square=True, linewidths = 0.05,vmax=1, cmap=cmap,center = 0.7) # vmax=1, vmin=0,
    
    plt.savefig("plot/hotfig2.png")
    plt.clf()

plothot2()

def BuildModel():
    #Using Step-wise Regression to Build a Robust Model
    X = df2[predictors]
    y = df2['spd70']

    X = sm.add_constant(X)
    #print(X.iloc[:5, :5])

    alpha = 0.05
    model = sm.OLS(y, X).fit()

    print(model.summary())
    return X,y

X, y = BuildModel()
X = X.drop('const', axis=1)

def train_split():
    # 选择 p > 0.05，说明独立性不好的，删掉。

    # 用回归模型做预报
    from sklearn.model_selection import train_test_split

    #移除常数项，因为sk-lean库不像statsmodels，会自动给我们添加一项
    # first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
    # X = X.drop('const', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_split()

def train_regress():
    from sklearn.linear_model import LinearRegression
    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(X_train, y_train)

    # make a prediction set using the test set
    prediction = regressor.predict(X_test)
    print(prediction)
    # res=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index),only_fcst],axis=1).dropna()
    res=pd.concat([y_test,pd.DataFrame({'end_var':prediction},index=y_test.index)],axis=1).dropna()
    print(res)

    # Evaluate the prediction accuracy of the model
    from sklearn.metrics import mean_absolute_error, median_absolute_error
    print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
    print("The Mean Absolute Error: %.2f m/s " % mean_absolute_error(y_test, prediction))
    print("The Median Absolute Error: %.2f m/s " % median_absolute_error(y_test, prediction))

train_regress()