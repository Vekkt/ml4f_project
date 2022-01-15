import bitfinex
import pandas as pd
import numpy as np
import datetime
import time



"""First 2 functions in this file are taken from :
https://medium.com/coinmonks/how-to-get-historical-crypto-currency-data-954062d40d2d"""


def fetch_data(start=1364767200000, stop=1545346740000, symbol='btcusd', interval='1m', tick_limit=1000, step=60000000):
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()

    data = []
    start = start - step
    while start < stop:
        try:

            start = start + step
            end = start + step
            res = api_v2.candles(symbol=symbol, interval=interval, limit=tick_limit, start=start, end=end)
            data.extend(res)
            print('Retrieving data from {} to {} for {}'.format(pd.to_datetime(start, unit='ms'),
                                                                pd.to_datetime(end, unit='ms'), symbol))
            time.sleep(60/90)
        except: pass
    return data

def get_data(ticker='BTCUSD', start=datetime.datetime(2016, 1, 1, 0, 0), end=datetime.datetime.now(), freq='1m'):
    #pair = 'BTCUSD' # What is the currency pair we are interested in
    bin_size = freq # This is the resolution at which we request the data
    limit = 1000 # How many data points per call are we asking for
    time_step = 1000 * 60 * limit # From the above calulate the size of each sub querry

    # Fill in the start and end time of interest and convert it to timestamps
    #t_start = datetime.datetime(2016, 1, 1, 0, 0)
    t_start = start
    t_start = time.mktime(t_start.timetuple()) * 1000

    #t_stop = datetime.datetime(2017, 1, 1, 23, 59)
    #t_stop = datetime.datetime.now()
    t_stop = end
    t_stop = time.mktime(t_stop.timetuple()) * 1000

    # Create an bitfinex_api instance
    api_v1 = bitfinex.bitfinex_v1.api_v1()

    # Collect the data
    pair_data = fetch_data(start=t_start, stop=t_stop, symbol=ticker, interval=bin_size, tick_limit=limit, step=time_step)

    ind = [np.ndim(x) != 0 for x in pair_data]
    pair_data = [i for (i, v) in zip(pair_data, ind) if v]

    # Create pandas data frame and clean data
    names = ['Timestamp', 'Open', 'Close', 'High', 'Low', 'Volume']
    df = pd.DataFrame(pair_data, columns=names)
    df.drop_duplicates(inplace=True)
    df.set_index('Timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    df.sort_index(inplace=True)
    df['Volume_(Currency)'] = df['Close'] * df['Volume']


    return df

def load_data(ticker, new_data=True):
    """Ticker's are like this;: 'BTCUSD', 'ETHUSD', 'XRPUSD'."""
    data = pd.read_csv(ticker+'_1m_2015_01_01_now.csv') #load the data from csv
    data.index = pd.to_datetime(data['Timestamp'])  #set time index
    #data.drop(labels='ret', axis=1, inplace=True)   #drop the ret column
    data.drop(labels='Timestamp', axis=1, inplace=True) # drop the time column
    #print(data.keys())
    data['ret'] = data['Close'].pct_change()
    if new_data:
        start = data.index.max() #identify the last timestamp and fetch data starting from here
        ext = get_data(ticker, start)
        data = pd.concat([data, ext], axis=0)   #conat the 2 frames
        data.drop_duplicates(inplace=True)  #get rid of duplicates
        data.sort_index(inplace=True)
        data.fillna(method='backfill', inplace=True)
        data['ret'] = data['Close'].pct_change()
        data.to_csv(ticker + '_1m_2015_01_01_now.csv')   #update the file so don't have to download a lot of data at a time
    else: pass
    return data


def perf_measure(real, pred):
    if all(i >= 0 for i in real.flatten()):  # if all values are above 0 transform the data into up and down movemenets
        real = real[1:] - real[:-1]
        pred = pred[1:] - pred[:-1]
    else:
        pass

    real_ = real.flatten() > 0
    pred_ = pred.flatten() > 0

    TP, FP = 0, 0
    TN, FN = 0, 0

    for i in range(len(pred_)):
        if real_[i] == pred_[i] == True:
            TP += 1
        if pred_[i] == True and real_[i] != pred_[i]:
            FP += 1
        if real_[i] == pred_[i] == False:
            TN += 1
        if pred_[i] == False and real_[i] != pred_[i]:
            FN += 1

    accuracy = (TP + TN) / len(real)
    f1 = TP / (TP + 0.5 * (FP + FN))

    return accuracy, f1




def prepare_data(data, freq='D'):
  # Volume and returns
  vol = data['Volume'].resample(freq).sum()
  
  #ret = data['Ret'].resample(freq).apply(lambda x: (x+1).product() - 1)
  log_ret = np.log(1 + data['ret']).resample(freq).sum()
  ret = (np.exp(log_ret) - 1).rename('ret')
  max = data['High'].resample(freq).max()
  min = data['Low'].resample(freq).min()
  range = (max - min).rename('Range')
  price = data['Close'].resample(freq).last()
  #ret = (100*price.pct_change()).rename('Ret')
  std = np.sqrt(24*60)*(data['ret'].resample(freq).std()).rename('Volatility')

  data = pd.concat([std, vol, ret, min, max, range, price], axis=1)
  data.dropna(inplace=True)
  return data



if __name__ == '__main__':
    data = load_data('XRPUSD')
