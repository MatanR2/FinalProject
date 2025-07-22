!pip install yfinance
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

start = '2020-09-29'
end = '2021-3-25'

# Fetch BTC data
btc = yf.download('BTC-USD', start=start, end=end)
btc.reset_index(inplace=True)
btc['Volatility (%)'] = ((btc['Close'] - btc['Open']) / btc['Open']) * 100
btc['Daily Change (indicator)'] = btc.loc[:, 'Volatility (%)']
btc.loc[btc['Daily Change (indicator)'] >= 0.25, 'Daily Change (indicator)'] = 1
btc.loc[btc['Daily Change (indicator)'] <= -0.25, 'Daily Change (indicator)'] = -1
btc.loc[(btc['Daily Change (indicator)'] > -0.25) & (btc['Daily Change (indicator)'] < 0.25), 'Daily Change (indicator)'] = 0
btc_filtered = btc[['Date', 'Open', 'Close', 'Volatility (%)', 'Daily Change (indicator)']]
btc_filtered.to_csv("Yahoo_Finance_Bitcoin.csv", index=False)


