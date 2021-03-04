  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

data=pd.read_csv("C:\\Users\\jhs473\\Downloads\\divesecbars.csv")

data['timestamp'] = (pd.to_datetime(data['epochtime'], unit='s')
                     .dt.tz_localize('utc')
                     )    
data
data['change']=data['close']-data['open']
data['percentchange']=data['change'] / data['open'] *100
ticker1=data[data.tickerid==1]
ticker2=data[data.tickerid==2]
ticker3=data[data.tickerid==3]
ticker4=data[data.tickerid==4]
ticker0=data[data.tickerid==0]

plt.plot(ticker0.close,linewidth=1)
plt.show()
plt.plot(ticker1.close,linewidth=1)
plt.show()
plt.plot(ticker2.close,linewidth=1)
plt.show()
plt.plot(ticker3.close,linewidth=1)
plt.show()
plt.plot(ticker4.close,linewidth=1)
plt.legend()
plt.show()

ticker1['weightedavgprice'].plot(xlabel="Time",ylabel="Weighted Average Price",figsize=(16,8),title='Weighted Average Price')
plt.show()


plt.plot(ticker0.epochtime,ticker0.weightedavgprice)
plt.title('Weighted Average Price Ticker 0',linewidth=1)
plt.xlabel('timestamp')
plt.ylabel('Weighted Average Price')
plt.show()


plt.plot(ticker1.epochtime,ticker1.weightedavgprice)
plt.title('Weighted Average Price Ticker 1',linewidth=1)
plt.xlabel('Epoch Time')
plt.ylabel('Weighted Average Price')
plt.show()

plt.plot(ticker2.epochtime,ticker2.weightedavgprice)
plt.title('Weighted Average Price Ticker 2',linewidth=1)
plt.xlabel('Epoch Time')
plt.ylabel('Weighted Average Price')
plt.show()

plt.plot(ticker3.epochtime,ticker3.weightedavgprice)
plt.title('Weighted Average Price Ticker 3',linewidth=1)
plt.xlabel('Epoch Time')
plt.ylabel('Weighted Average Price')
plt.show()

plt.plot(ticker4.epochtime,ticker4.weightedavgprice)
plt.title('Weighted Average Price Ticker 4',linewidth=1)
plt.xlabel('Epoch Time')
plt.ylabel('Weighted Average Price')
plt.show()


plt.plot(ticker4.epochtime,ticker4.volume,linewidth=.1)
plt.plot(ticker2.epochtime,ticker2.volume,linewidth=.1)
plt.plot(ticker1.epochtime,ticker1.volume,linewidth=.1)
plt.plot(ticker3.epochtime,ticker3.volume,linewidth=.1)
plt.plot(ticker0.epochtime,ticker0.volume,linewidth=.1)
plt.yscale('log')
plt.show()

plt.plot(ticker4.epochtime,ticker4.change,linewidth=.1)
plt.plot(ticker2.epochtime,ticker2.change,linewidth=.1)
plt.plot(ticker1.epochtime,ticker1.change,linewidth=.1)
plt.plot(ticker3.epochtime,ticker3.change,linewidth=.1)
plt.plot(ticker0.epochtime,ticker0.change,linewidth=.1)
plt.show()

plt.plot(ticker4.epochtime,ticker4.percentchange,linewidth=.1)
plt.plot(ticker2.epochtime,ticker2.percentchange,linewidth=.1)
plt.plot(ticker1.epochtime,ticker1.percentchange,linewidth=.1)
plt.plot(ticker3.epochtime,ticker3.percentchange,linewidth=.1)
##plt.plot(ticker0.epochtime,ticker0.percentchange,linewidth=.1)
plt.show()

