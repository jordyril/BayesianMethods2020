"""
"""
import os

import pandas as pd
import numpy as np

from MyUtils.dataprocessing import DataProcessor

# -------------------------------------
# READ IN/CLEAN/TRANSFoRM TO RETURNS DATA
# -------------------------------------
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")
dataprocessor = DataProcessor()


# INDEXES
index_data = dataprocessor.read_csv(
    "indexes", index_col=0, header=[0, 1], parse_dates=True
)

returns = pd.DataFrame()
for i in index_data["close"]:
    subdata = index_data["close"][i]
    # temp = (subdata  - subdata .shift()) / subdata .shift()
    temp = np.log(subdata).diff()  # log-returns
    returns[i] = temp.dropna()

# CURRENCIES
currency_data = dataprocessor.read_csv(
    "currencies", index_col=0, header=[0, 1], parse_dates=True
)

currencies = pd.DataFrame()
for cur in currency_data.columns.levels[0]:
    subdata = currency_data[cur]["close"]
    # temp = (subdata  - subdata .shift()) / subdata .shift()
    temp = np.log(subdata).diff()  # log-returns
    currencies[cur] = temp.dropna()

# -------------------------------------
# SELECT/SAVE DATA
# -------------------------------------
ind = "bel20"
index_returns = returns[ind].dropna() * 100
dataprocessor.to_csv(index_returns, "index_returns")
index = index_data["close"][ind].dropna()
dataprocessor.to_csv(index, "index")

cur = "EURCHF"
currency_returns = currencies[cur].dropna() * 100
dataprocessor.to_csv(currency_returns, "currency_returns")
currency = currency_data[cur]["close"].dropna()
dataprocessor.to_csv(currency, "currency")
