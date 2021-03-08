import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
import time
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist



#NOTE: I am mannually assigning the path because my Visual Basic Code had a different working directory
# if your working directory is same as the python script's path, you can comment this out.
path=os.getcwd()#+'/Group assignment/BZAN545_Group6_Assignment1/'    


#-----------------------------------------DATE EXPLORATION AND CLEANING--------------------------------------------------------------------------
def explore_and_clean_data():
    #PURPOSE:  function will read the csv file, explore the data, clean if necessary, and return cleaned data
    #INPUT: None
    #OUTPUT: cleaned dataframe

    #comment here for any anomaly in the data that needs cleaning


    #----------------------------------------


    print("NOTE: This file is for data exploration and cleaning")

    raw_data = pd.read_csv(path+'/fivesecbars.csv')
    
    #view the data types
    print('\n\nView data types by column \n',raw_data.dtypes)

    #convert epoch time to panda datetime
    print('\n\nCreate a column named datetime to parse epochtime column into standard pandas datetime format')
    raw_data['datetime']=pd.to_datetime(raw_data['epochtime'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    print('\nDropping epochtime column')
    raw_data.drop(columns=['epochtime'], inplace=True)

    print('\n\nView first five rows \n',raw_data.head())

 
    #explore mising values by columns
    print('\n\nMissing values by columns \n',raw_data.isna().sum())

    #summary of columns
    print('\n\nSummary of columns \n',raw_data.describe())

    #after cleaning return clean data
    clean_data=raw_data
    
    return clean_data

#-------------------------------------------------VISUALIZATION------------------------------------------------------------------
def visualize(data):
    #PURPOSE:  visualize key variables
    #INPUT: data: cleaned data from explore_and_clean_data()
    #OUTPUT: None

    print("NOTE: This file is for data visualization")

    #sample plot
      



    import numpy as np
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import plotly.graph_objects as go
    data['change']=data['close']-data['open']
    data['percentchange']=data['change'] / data['open'] *100

    data['time'] = pd.to_datetime(data.datetime)
    ticker1=data[data.tickerid==1]
    ticker2=data[data.tickerid==2]
    ticker3=data[data.tickerid==3]
    ticker4=data[data.tickerid==4]
    ticker0=data[data.tickerid==0]
    
    ##breaking data into 5 minute intervals for visulization purposes for each stock
    averages1=ticker1.groupby(pd.Grouper(key='time',freq='5min')).mean()
    sums1=ticker1.groupby(pd.Grouper(key='time',freq='5min')).sum()
    min1=ticker1.groupby(pd.Grouper(key='time',freq='5min')).min()
    max1=ticker1.groupby(pd.Grouper(key='time',freq='5min')).max()

    averages0=ticker0.groupby(pd.Grouper(key='time',freq='5min')).mean()
    sums0=ticker0.groupby(pd.Grouper(key='time',freq='5min')).sum()
    min0=ticker0.groupby(pd.Grouper(key='time',freq='5min')).min()
    max0=ticker0.groupby(pd.Grouper(key='time',freq='5min')).max()

    
    averages2=ticker2.groupby(pd.Grouper(key='time',freq='5min')).mean()
    sums2=ticker2.groupby(pd.Grouper(key='time',freq='5min')).sum()
    min2=ticker2.groupby(pd.Grouper(key='time',freq='5min')).min()
    max2=ticker2.groupby(pd.Grouper(key='time',freq='5min')).max()

    averages3=ticker3.groupby(pd.Grouper(key='time',freq='5min')).mean()
    sums3=ticker3.groupby(pd.Grouper(key='time',freq='5min')).sum()
    min3=ticker3.groupby(pd.Grouper(key='time',freq='5min')).min()
    max3=ticker3.groupby(pd.Grouper(key='time',freq='5min')).max()

    averages4=ticker4.groupby(pd.Grouper(key='time',freq='5min')).mean()
    sums4=ticker4.groupby(pd.Grouper(key='time',freq='5min')).sum()
    min4=ticker4.groupby(pd.Grouper(key='time',freq='5min')).min()
    max4=ticker4.groupby(pd.Grouper(key='time',freq='5min')).max()

    ##candlestick plots for each stock 0,1,2,3,4
    ##stock 0
    import plotly.graph_objects as go
    averages0['newdates']=pd.to_datetime(averages0.index.values)
    fig = go.Figure(data=[go.Candlestick( x=averages0['newdates'],
                open=averages0['open'], high=max0['high'],
                low=min0['low'], close=averages0['close'])
                     ])

    fig.update_layout(title='Candlestick Chart Stock 0',xaxis_rangeslider_visible=True)
    fig.show()

    ##stock 1 candlestick
    averages1['newdates']=pd.to_datetime(averages1.index.values)
    fig = go.Figure(data=[go.Candlestick( x=averages1['newdates'],
                    open=averages1['open'], high=max1['high'],
                    low=min1['low'], close=averages1['close'])
                        ])

    fig.update_layout(title='Candlestick Chart Stock 1',xaxis_rangeslider_visible=True)
    fig.show() 
    ## stock 2 candlestick
    averages2['newdates']=pd.to_datetime(averages2.index.values)
    fig = go.Figure(data=[go.Candlestick( x=averages2['newdates'],
                open=averages2['open'], high=max2['high'],
                low=min2['low'], close=averages2['close'])
                     ])

    fig.update_layout(title='Candlestick Chart Stock 2',xaxis_rangeslider_visible=True)
    fig.show()

    ##stock 3 candlestick

    averages3['newdates']=pd.to_datetime(averages3.index.values)
    fig = go.Figure(data=[go.Candlestick( x=averages3['newdates'],
                open=averages3['open'], high=max3['high'],
                low=min3['low'], close=averages3['close'])
                     ])

    fig.update_layout(title='Candlestick Chart Stock 3',xaxis_rangeslider_visible=True)
    fig.show()
    #stock 4 candlestick
    averages4['newdates']=pd.to_datetime(averages4.index.values)
    fig = go.Figure(data=[go.Candlestick( x=averages4['newdates'],
                open=averages4['open'], high=max4['high'],
                low=min4['low'], close=averages4['close'])
                     ])

    fig.update_layout(title='Candlestick Chart Stock 4',xaxis_rangeslider_visible=True)
    fig.show()

    ####stacked histogram log y scale, percent change all stocks
    ##use this
    plt.hist([sums0['percentchange'],sums1['percentchange'],sums2['percentchange'],sums3['percentchange'],sums4['percentchange']], stacked=True,bins=50,edgecolor='black',align='left')
    plt.title('Stacked Histogram Percent Change (5 minute increments)')
    plt.yscale('log')
    plt.ylabel('Frequency (log scale)')
    plt.xlabel('Percent Change')
    plt.legend(["Stock 0", "Stock 1","Stock 2","Stock 3","Stock 4"])
    plt.show()

    ##percent change each stock same graph, only data 1/29 12pm-1/30 12am
    ## this is where bulk of data lies so I trimmed it to not include begininng and end
    ## probably the one we should use
    plt.plot(sums0.percentchange[195:346],linestyle='solid',label='Stock 0',linewidth=0.3)
    plt.plot(sums1.percentchange[195:346],linestyle='solid',label='Stock 1',linewidth=0.3)
    plt.plot(sums2.percentchange[195:346],linestyle='solid',label='Stock 2',linewidth=0.3)
    plt.plot(sums3.percentchange[195:346],linestyle='solid',label='Stock 3',linewidth=0.3)
    plt.plot(sums4.percentchange[195:346],linestyle='solid',label='Stock 4',linewidth=0.3)
    plt.title('Percent Changes (1/29 12pm - 1/30 12am)')
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.show()

    ##percent change each stock same graph, doesnt look good as plot above
    ##this is the one with all data points except long right tail of stock 0
    ##maybe use this one

    plt.plot(sums0.percentchange[0:346],linestyle='solid',label='Stock 0',linewidth=0.3)
    plt.plot(sums1.percentchange,linestyle='solid',label='Stock 1',linewidth=0.3)
    plt.plot(sums2.percentchange,linestyle='solid',label='Stock 2',linewidth=0.3)
    plt.plot(sums3.percentchange,linestyle='solid',label='Stock 3',linewidth=0.3)
    plt.plot(sums4.percentchange,linestyle='solid',label='Stock 4',linewidth=0.3)
    plt.title('Percent Changes (1/28 8pm - 1/30 12am)')
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.show()


    ##percent change each stock same graph, all data is here, doesnt look good with stock 0's long right tail
    ##this is the one with all data points
    ##probably shouldnt use this one
    plt.plot(sums0.percentchange,linestyle='solid',label='Stock 0',linewidth=0.3)
    plt.plot(sums1.percentchange,linestyle='solid',label='Stock 1',linewidth=0.3)
    plt.plot(sums2.percentchange,linestyle='solid',label='Stock 2',linewidth=0.3)
    plt.plot(sums3.percentchange,linestyle='solid',label='Stock 3',linewidth=0.3)
    plt.plot(sums4.percentchange,linestyle='solid',label='Stock 4',linewidth=0.3)
    plt.title('Percent Changes (1/28 8pm - 2/4 9pm)')
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.show()


    ##volume each stock same graph, only data 1/29 12pm-1/30 12am
    ## this is where bulk of data lies so I trimmed it to not include begininng and end
    ## maybe the one we should use
    plt.plot(sums0.volume[195:346],linestyle='solid',label='Stock 0',linewidth=0.5)
    plt.plot(sums1.volume[195:346],linestyle='solid',label='Stock 1',linewidth=0.5)
    plt.plot(sums2.volume[195:346],linestyle='solid',label='Stock 2',linewidth=0.5)
    plt.plot(sums3.volume[195:346],linestyle='solid',label='Stock 3',linewidth=0.5)
    plt.plot(sums4.volume[195:346],linestyle='solid',label='Stock 4',linewidth=0.5)
    plt.title('Volume (1/29 12pm - 1/30 12am)')
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()

    ##volume each stock same graph, doesnt look good as plot above but may be better
    ##this is the one with all data points except long right tail of stock 0
    ##maybe/probably use this one

    plt.plot(sums0.volume[0:346],linestyle='solid',label='Stock 0',linewidth=0.5)
    plt.plot(sums1.volume,linestyle='solid',label='Stock 1',linewidth=0.5)
    plt.plot(sums2.volume,linestyle='solid',label='Stock 2',linewidth=0.5)
    plt.plot(sums3.volume,linestyle='solid',label='Stock 3',linewidth=0.5)
    plt.plot(sums4.volume,linestyle='solid',label='Stock 4',linewidth=0.5)
    plt.title('Volume (1/28 8pm - 1/30 12am)')
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()


    ##volume each stock same graph, all data is here, doesnt look good with stock 0's long right tail
    ##this is the one with all data points
    ##looks really bad
    plt.plot(sums0.volume,linestyle='solid',label='Stock 0',linewidth=0.3)
    plt.plot(sums1.volume,linestyle='solid',label='Stock 1',linewidth=0.3)
    plt.plot(sums2.volume,linestyle='solid',label='Stock 2',linewidth=0.3)
    plt.plot(sums3.volume,linestyle='solid',label='Stock 3',linewidth=0.3)
    plt.plot(sums4.volume,linestyle='solid',label='Stock 4',linewidth=0.3)
    plt.title('Volume (1/28 8pm - 2/4 9pm)')
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()

    #K-means Clustering
    #transform the data to begin cluster analysis
    #Create a PCA model to reduce our data to 2 dimensions for visualisation
    clean_data=data
    pca=PCA(2)
    df=pca.fit_transform(clean_data)
    df.shape

    #Elbow Method to estimate how many clusters 
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 15)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(df)
        kmeanModel.fit(df)

        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / df.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(df, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / df.shape[0]
        mapping2[k] = kmeanModel.inertia_

    #Distortion values Table where distortion is the sum of square errors (SSE)

    for key, val in mapping1.items():
    print(f'{key} : {val}')

    return None

    #plot of elbow method using Distortion
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

    #table of Inertia results where Inertia tells us how far the points within a cluster are
    for key, val in mapping2.items():
    print(f'{key} : {val}')




#-------------------------------------------------MODEL------------------------------------------------------------------    
def model(data):
    #PURPOSE: use ML to predict stock price
    #INPUT: data: cleaned data from explore_and_clean_data()
    #OUTPUT: None

    print("NOTE: This file is for predicting stock price")
    new_data=data.copy()
    #drop datetime variable
    new_data=new_data.drop(columns=['datetime'])
    
        #divide the dataset into features and response variables**
    X = new_data.drop('close', axis='columns')
    Y = new_data['close']

    X.head()

    Y.head()
    
        #Split into training and testing data# use 80% for training and 20% for testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, random_state=0)

        #get the size of the traiing data
    X_train.shape

    X_test.shape

        #Data preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.compose import make_column_transformer

        #define the model class
    model = LinearRegression()
    feature_scaling = StandardScaler()   #for feature scaling 


        #scale the training and testing features 
    X_train_scaled = feature_scaling.fit_transform(X_train)
    X_test_scaled = feature_scaling.transform(X_test)


        #fit the model
    model.fit(X_train, Y_train)

        #make prediction on test dataset
    Y_pred = model.predict(X_test_scaled)

        #view the predictied Y
    Y_pred

        #import metrics
    from sklearn import metrics

        #get model intercept, coefficient, MAE, MSE, R squared 
    model.intercept_

    model.coef_

        #estimate the mean absolute error
    metrics.mean_absolute_error(Y_test, Y_pred)

        #get the mean square error
    metrics.mean_squared_error(Y_test, Y_pred)

        #calculate the r-squared value
    r2_value = metrics.r2_score(Y_test,Y_pred)
    print('r-squared value: ',r2_value)
    
    
    #using cross validation
    from sklearn.model_selection import cross_val_score

        #try the 10 fold cross validated regression
    scores = cross_val_score(model, X_train, Y_train, cv=10)
    print('\nCross validation scores: ',scores)


    return None



#---------------------------------------------------MAIN FUNCTION--------------------------------------------------



def main():
    
    #call function that imports data, describes it, cleans (if necessary), and return cleaned dataframe
    data=explore_and_clean_data()
    print(type(data))


    #call function to visualize data
    visualize(data)

    #call function to predict stock price
    model(data)


if __name__ == "__main__":
    main()


