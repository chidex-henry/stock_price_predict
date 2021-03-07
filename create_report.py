import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
import time
import os


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
      



    data['change']=data['close']-data['open']
    data['percentchange']=data['change'] / data['open'] *100
    ticker1=data[data.tickerid==1]
    ticker2=data[data.tickerid==2]
    ticker3=data[data.tickerid==3]
    ticker4=data[data.tickerid==4]
    ticker0=data[data.tickerid==0]

    ##weighted avg price vs time ticker 0
    x=ticker0.weightedavgprice
    y=ticker0.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Weighted Average Price Ticker 0')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Price')
    plt.show()

    ##weighted avg price vs time ticker 1
    x=ticker1.weightedavgprice
    y=ticker1.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Weighted Average Price Ticker 1')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Price')
    plt.show()

    ##weighted avg price vs time ticker 2
    x=ticker2.weightedavgprice
    y=ticker2.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Weighted Average Price Ticker 2')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Price')
    plt.show()

    ##weighted avg price vs time ticker 3
    x=ticker3.weightedavgprice
    y=ticker3.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Weighted Average Price Ticker 3')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Price')
    plt.show()

    ##big visual tickers 1,2,3,4 ## doesnt look very nice
    x=ticker1.weightedavgprice
    y=ticker1.datetime
    x1=ticker2.weightedavgprice
    y1=ticker2.datetime
    x2=ticker3.weightedavgprice
    y2=ticker3.datetime
    x3=ticker4.weightedavgprice
    y3=ticker4.datetime


    dates=matplotlib.dates.date2num(y)
    dates1=matplotlib.dates.date2num(y1)
    dates2=matplotlib.dates.date2num(y2)
    dates3=matplotlib.dates.date2num(y3)


    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.plot_date(dates1,x1,linestyle='solid',linewidth=0.2,markersize=0)
    plt.plot_date(dates2,x2,linestyle='solid',linewidth=0.2,markersize=0)
    plt.plot_date(dates3,x3,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Weighted Average Price')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Price')
    plt.show()

    ##big visual tickers 3,4 ## look very nice

    x2=ticker3.weightedavgprice
    y2=ticker3.datetime
    x3=ticker4.weightedavgprice
    y3=ticker4.datetime



    dates2=matplotlib.dates.date2num(y2)
    dates3=matplotlib.dates.date2num(y3)



    plt.plot_date(dates2,x2,linestyle='solid',linewidth=0.2,markersize=0)
    plt.plot_date(dates3,x3,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Weighted Average Price')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Price')
    plt.show()
    ##percent change vs time ticker 0
    x=ticker0.percentchange
    y=ticker0.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Percent Change Ticker 0')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.show()

    ##percent change vs time ticker 1
    x=ticker1.percentchange
    y=ticker1.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Percent Change Ticker 1')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.show()

    ##percent change vs time ticker 2
    x=ticker2.percentchange
    y=ticker2.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Percent Change Ticker 2')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.show()

    ##percent change vs time ticker 3
    x=ticker3.percentchange
    y=ticker3.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Percent Change Ticker 3')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.show()

    ##percent change vs time ticker 4
    x=ticker4.percentchange
    y=ticker4.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Percent Change Ticker 4')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.show()



    ##percent change vs time ticker 4
    x=ticker4.change
    y=ticker4.datetime


    dates=matplotlib.dates.date2num(y)
    plt.plot_date(dates,x,linestyle='solid',linewidth=0.2,markersize=0)
    plt.title('Change Ticker 4')
    plt.xlabel('Time')
    plt.ylabel('Change')
    plt.show()

    ##percent change histogram Ticker 0
    ticker0['percentchange'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 0 Percent Change Histogram')
    plt.xlabel('Percent Change')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker0.percentchange.describe()

    ##percent change histogram Ticker 1
    ticker1['percentchange'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 1 Percent Change Histogram')
    plt.xlabel('Percent Change')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker1.percentchange.describe()

    ##percent change histogram Ticker 2
    ticker2['percentchange'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 2 Percent Change Histogram')
    plt.xlabel('Percent Change')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker2.percentchange.describe()

    ##percent change histogram Ticker 3
    ticker3['percentchange'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 3 Percent Change Histogram')
    plt.xlabel('Percent Change')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker3.percentchange.describe()

    ##percent change histogram Ticker 4
    ticker4['percentchange'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 4 Percent Change Histogram')
    plt.xlabel('Percent Change')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker4.percentchange.describe()

    ##volume histogram Ticker 0
    ticker0['volume'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 0 Volume Histogram')
    plt.xlabel('Volume')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker0.volume.describe()

    ##volume histogram Ticker 1
    ticker1['volume'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 1 Volume Histogram')
    plt.xlabel('Volume')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker1.volume.describe()

    ##volume histogram Ticker 2
    ticker2['volume'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 2 Volume Histogram')
    plt.xlabel('Volume')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker2.volume.describe()

    ##volume histogram Ticker 3
    ticker3['volume'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 3 Volume Histogram')
    plt.xlabel('Volume')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker3.volume.describe()





    ##volume histogram Ticker 4
    ticker4['volume'].hist(bins = 50, figsize = (10,5)) 
    plt.title('Ticker 4 Volume Histogram')
    plt.xlabel('Volume')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.show()
    #satistics
    ticker4.volume.describe()


    return None




#-------------------------------------------------MODEL------------------------------------------------------------------    
def model(data):
    #PURPOSE: use ML to predict stock price
    #INPUT: data: cleaned data from explore_and_clean_data()
    #OUTPUT: None

    print("NOTE: This file is for predicting stock price")

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


