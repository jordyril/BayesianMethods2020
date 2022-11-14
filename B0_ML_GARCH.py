"""
"""
import os

import pandas as pd
import numpy as np
import scipy.stats as scs
import pymc3 as pm
from statsmodels.tsa.arima_model import ARMA
from arch import arch_model

from MyUtils.dataprocessing import DataOutputting

random_state = 13
np.random.seed(random_state)

# ==================================================
# READ IN DATA
# ==================================================
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

from A3_DataImport import prior_demeaned, prior

output = DataOutputting()

latex_dir = "../Latex/"

# ------------------------------------
# Establishing prior for ARMA models
# ------------------------------------
def run_arma():
    arma_results = {}
    no_res = []
    for a in range(0, 6):
        for m in range(0, a + 1):
            arma_res = ARMA(prior, (a, m)).fit(transparams=False, disp=False)
            arma_results[f"ARMA_{a}_{m}"] = arma_res

    output.save_to_pickle("prior_arma_results", arma_results)
    return


# ------------------------------------
# Establishing prior for GARCH models
# ------------------------------------
garch11_ml_results = arch_model(prior_demeaned, p=1, q=1).fit(update_freq=100)
vol_table = garch11_ml_results.summary().tables[2]
vol_table.title = None


if __name__ == "__main__":
    # print(vol_table.as_latex_tabular())
    for par in ["omega", "alpha[1]", "beta[1]"]:
        print(
            f"{par} - {garch11_ml_results.params[par]} - {garch11_ml_results.std_err[par]} - {garch11_ml_results.std_err[par] * 5}"
        )

    run_arma()
