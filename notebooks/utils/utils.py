import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

def rename_columns_and_format(df):
    '''
    Recibe un dataframe con las columnas Date, Open, High, Low, Close y Volume
    y retorna uno con las mismas columnas renombradas, ordenado por fecha y 
    con fecha en formato YYYmmdd HMS
    '''
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
    '''
    Realiza una particion para entrenamiento y testeo para los inputs y el output del modelo.
    Recibe el porcentaje de datos para entrenamiento, el input y el output del modelo.
    Retorna los inputs de train y test, y los outputs de train y test.
    '''
    train_size = int(len(x)*p_train)

    x_train = x[0:train_size]
    x_test = x[train_size:]

    y_train = y[0:train_size]
    y_test = y[train_size:]
    
    return x_train, x_test, y_train, y_test



def create_windowed_dataset(df_prices, target, window):
    '''
    Retorna un dataframe en forma de ventana deslizante para entrenamiento y/o testeo de un modelo.
    
    Recibe un dataframe compuesto por lo precios y/o indicadores de un activo, los valores de salida
    para cada uno de esas filas del primer dataset y el tamaño de la ventana (dias).
    '''
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
    Recibe un dataframe con las columnas Date, Open, High, Low, Close y Volume de un activo
    y retorna el mismo dataset con mas columnas correspondientes a mas indicadores financieros
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

def plot(x1, x2, title, xlabel, ylabel, legend):
    '''
    Plotea un grafico correspondiente a dos variables
    recibe las dos variables a graficar, un titulo, label para el eje x, label para el eje y, y una leyenda.
    '''
    plt.figure(figsize=(13,6))
    plt.plot(x1)
    plt.plot(x2)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.legend(legend, loc='upper right')
    plt.show()

def get_model(x_input, y_input):
    '''
    Retorna un modelo de redes neuronales con 
    '''
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, (x_input, 3), input_shape=(x_input, y_input,1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1, 2)))
    
    model.add(tf.keras.layers.Conv2D(32, (1, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1, 2)))

    model.add(tf.keras.layers.Conv2D(32, (1, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1, 2)))
    
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    #model.summary()
    
    return model