import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#-----------------------------------------DATE EXPLORATION AND CLEANING--------------------------------------------------------------------------
def explore_and_clean_data():
    #PURPOSE:  function will read the csv file, explore the data, clean if necessary, and return cleaned data
    #INPUT: None
    #OUTPUT: cleaned dataframe

    print("NOTE: This file is for data exploration and cleaning")

    raw_data = pd.read_csv(r'fivesecbars.csv')
    # print(raw_data.describe())

    # #convert epoch time to panda datetime
    # raw_data['datetime_pd']=pd.to_datetime(raw_data['epochtime'])

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
    sns_plot =sns.scatterplot(data=data, x='high',y='low')
    plt.show()
    sns_plot.figure.savefig(r'plots/sample_plot.png')


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


