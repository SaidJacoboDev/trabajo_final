import pandas as pd
import numpy as np
import talib

def rename_columns_and_format(df):
    df.rename(columns={'Gmt time' : 'date', 
                   'Open':'open',
                   'High':'high',
                   'Low':'low',
                   'Close':'close',
                   'Volume':'volume'}, inplace=True)

    df["date"] = pd.to_datetime(df['date'])
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df.sort_values('date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    return df

def train_test_split(p_train, x, y):
    train_size = int(len(x)*p_train)

    x_train = x[0:train_size]
    x_test = x[train_size:]

    y_train = y[0:train_size]
    y_test = y[train_size:]
    
    return x_train, x_test, y_train, y_test



def create_windowed_dataset(df_prices, target, window):
    prices = df_prices.to_numpy().T
    
    x = []
    y = np.empty([0,1])

    if len(prices.shape) == 1:
        prices = np.expand_dims(prices, axis=0)
        
    for j in range(prices.shape[1]-window):
        x_aux = []

        for i in range(prices.shape[0]):
            x_aux.append(np.array(prices[i][j:j+window]))

        y = np.append(y, target[j+window])
        x.append(x_aux)

    x = np.array(x)
    
    return x, y



def get_all_indicators(dataframe):
    '''
        Que hace, que recibe como entrada y que devuelve como salida
    '''
    df = dataframe.copy()    
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["ema_12"] = talib.EMA(df["close"], timeperiod=12)
    df["ema_26"] = talib.EMA(df["close"], timeperiod=26) 

    upper_band, middle_band, lower_band = talib.BBANDS(df["close"], timeperiod=20, 
                                                nbdevup=2, nbdevdn=2, matype=0)

    df["upper_bband"] = upper_band
    df["middle_bband"] = middle_band
    df["lower_bband"] = lower_band


    macd, macd_signal, macd_hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

    df["macd"] = macd 
    df["macd_signal"] = macd_signal 
    df["macd_hist"] = macd_hist
    
    k, d = talib.STOCH(df["high"], df["low"], df["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    df["k"] = k
    df["d"] = d
    
    return df
