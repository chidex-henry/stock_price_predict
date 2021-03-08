import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    import pandas as pd
    import numpy as np
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
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


