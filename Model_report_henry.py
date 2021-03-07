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

    #raw_data = pd.read_csv(path+'/fivesecbars.csv')
    raw_data = pd.read_csv('fivesecbars.csv')
    
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
    
    clean_data.head()
    
    return clean_data

#-------------------------------------------------VISUALIZATION------------------------------------------------------------------
def visualize(data):
    #PURPOSE:  visualize key variables
    #INPUT: data: cleaned data from explore_and_clean_data()
    #OUTPUT: None

    print("NOTE: This file is for data visualization")

    #sample plot
    sns_plot =sns.scatterplot(data=data, x='high',y='low')
    plt.show()
    sns_plot.figure.savefig(r'plots/sample_plot.png')


    return None




#-------------------------------------------------MODEL------------------------------------------------------------------    
def model(data):
    
    #import file from directory
 clean_data.head()

    # #drop epochtime column
new_data =clean_data.drop('epochtime', axis='columns')
new_data.head()
   
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
   

    #PURPOSE: use ML to predict stock price
    #INPUT: data: cleaned data from explore_and_clean_data()
    #OUTPUT: None

   # print("NOTE: This file is for predicting stock price")
    #print(data.shape)
    return None



#---------------------------------------------------MAIN FUNCTION--------------------------------------------------



def main():
    
    #call function that imports data, describes it, cleans (if necessary), and return cleaned dataframe
    data=explore_and_clean_data()
    print(type(data))


    #call function to visualize data
    visualize(data)

    #call function to predict stock price
    #model(data)
    model = LinearRegression()
    model(new_data)


if __name__ == "__main__":
    main()


