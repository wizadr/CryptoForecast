import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import requests
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import streamlit as st
import tensorflow as tf
import warnings
from tensorflow.keras.models import model_from_json
warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
        # Crypto Forecast

        """
         )
crypto=st.sidebar.selectbox('Select Choice of Crypto using their abbreviation',('BTC','ETH','DASH','DOGE','LTC','USDT'))
fiat=st.sidebar.selectbox('Select Choice of currency using their abbreviation',('INR','USD','CAD','EUR'))
limit = st.sidebar.number_input('Insert a timeframe between 0 and 2000', min_value=1,max_value=2000)

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym='+crypto+'&tsym='+fiat+'&limit='+str(limit))
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist=hist[hist['close']!=0]
hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)
@st.cache(allow_output_mutation=True)
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(hist, test_size=0.2)
#data=pd.DataFrame((train.close,test.close), columns=['train','test'])
data1={
    'train': train.close,
    'test': test.close
}
data=pd.DataFrame(data1)
st.line_chart(data)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(14, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_xlabel(crypto, fontsize=14)
    ax.set_ylabel(fiat, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show()

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (df.max() - df.min())

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


