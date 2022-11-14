"""
Collecting data

https://pandas-datareader.readthedocs.io/en/latest/readers/stooq.html

https://stooq.com/q/?s=^bel20

https://alpha-vantage.readthedocs.io/en/latest/

"""
import os
import time

import pandas as pd
import pandas_datareader
from alpha_vantage.foreignexchange import ForeignExchange
from tqdm import tqdm

from MyUtils.dataprocessing import DataProcessor

# -------------------------------------
# OPTIONS
# -------------------------------------
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

dataprocessor = DataProcessor()

# set data range (currencies max 10y)
start = "01-01-2000"  # M-D-Y
end = "17-06-2020"  # M-D-Y

# indices
index_list = ["^BEL20", "^AEX", "^SPY", "^CAC", "^OMXS", "^SMI", "^DAX"]

# Currencies
currency_list = [
    "EURUSD",
    "EURCHF",
    "USDCHF",
    "GBPCHF",
    "GBPUSD",
    # "GBPEUR",
]

# -------------------------------------
# DOWNLOAD data
# -------------------------------------
# COUNTRY INDICES- downloading from stooq.com
index_data = pandas_datareader.stooq.StooqDailyReader(
    symbols=index_list, start=start, end=end
).read()

index_data = index_data.sort_index()

index_data.columns = index_data.columns.set_levels(
    [x.lower() for x in index_data.columns.levels[0]], level=0
)

index_data.columns = index_data.columns.set_levels(
    [x[1:].lower() for x in index_data.columns.levels[1]], level=1
)

# CURRENCIES
cc = ForeignExchange(key=os.getenv("ALPHAVANTAGE_API_BF"))
currency_data = {}
i = 0
for pair in tqdm(currency_list):
    if i % 5 == 0:
        time.sleep(60)
    json = cc.get_currency_exchange_daily(
        from_symbol=pair[:3], to_symbol=pair[3:], outputsize="full"
    )[0]
    df = pd.DataFrame(json).T
    df.columns = ["open", "high", "low", "close"]
    currency_data[pair] = df
    i += 1

currency_data = pd.concat(
    list(currency_data.values()), keys=currency_data.keys(), axis=1
)

currency_data = currency_data.dropna()  # simply dropping days
currency_data = currency_data.astype(float)

currency_data.index = pd.to_datetime(currency_data.index)
currency_data = currency_data.sort_index()
currency_data = currency_data[start:end]
# -------------------------------------
# Save data
# -------------------------------------
dataprocessor.to_csv(index_data, "indexes")
dataprocessor.to_csv(currency_data, "currencies")

