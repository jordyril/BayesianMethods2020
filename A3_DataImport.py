"""
"""

import os
import numpy as np
import math
from MyUtils.dataprocessing import DataProcessor

# -------------------------------------
# READ IN PROCESSED DATA
# -------------------------------------
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")
dataprocessor = DataProcessor()

index = dataprocessor.read_csv("index", index_col=0, parse_dates=True)
index_returns = dataprocessor.read_csv("index_returns", index_col=0, parse_dates=True)

currency = dataprocessor.read_csv("currency", index_col=0, parse_dates=True)
currency_returns = dataprocessor.read_csv(
    "currency_returns", index_col=0, parse_dates=True
)
# -------------------------------------
# PRIOR - TRAIN - TEST
# -------------------------------------
prior_det, train = 0.15, 0.75

# INDEX
T = len(index_returns)
n_prior, n_train = (math.ceil(prior_det * T), math.ceil(train * T))

index_prior = index_returns[:n_prior]
index_train = index_returns[n_prior : n_prior + n_train]
index_test = index_returns[n_prior + n_train :]

index_samples = {"prior": index_prior, "train": index_train, "test": index_test}
# Currency
T = len(currency_returns)
n_prior, n_train = (math.ceil(prior_det * T), math.ceil(train * T))

currency_prior = currency_returns[:n_prior]
currency_train = currency_returns[n_prior : n_prior + n_train]
currency_test = currency_returns[n_prior + n_train :]

currency_samples = {
    "prior": currency_prior,
    "train": currency_train,
    "test": currency_test,
}

# -------------------------------------
# DATA PROCESSING GARCH
# -------------------------------------
prior, train, test = (
    index_samples["prior"],
    index_samples["train"],
    index_samples["test"],
)
T1, T = len(prior), len(train)
prior_demeaned = prior - prior.mean()
train_demeaned = train - train.mean()
test_demeaned = test - test.mean()
