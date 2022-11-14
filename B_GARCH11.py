"""
BASED ON CH11 in 'Bayesian methhods in Finance 
"""
import os

import pandas as pd
import numpy as np
import scipy.stats as scs
import pymc3 as pm
import theano

theano.config.cxx = ""
from arch import arch_model

from MyUtils.dataprocessing import DataOutputting

random_state = 13
np.random.seed(random_state)
# ==================================================
# READ IN DATA
# ==================================================
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

from A3_DataImport import train_demeaned, prior_demeaned
from B0_ML_GARCH import garch11_ml_results as prior_res

output = DataOutputting()

latex_dir = "../Latex/"
# ==================================================
# CREATE MODEL: GARCH(1, 1)
# ==================================================
# general
name = "garch11"

# ------------------
# TRAINING
# ------------------
# model specification
# def create_model():
garch11 = pm.Model()
with garch11:
    # data
    data = pm.Data("data", train_demeaned.T.values[0])
    # priors volatility process
    omega = pm.TruncatedNormal(
        "omega",
        mu=prior_res.params["omega"],
        sigma=5 * prior_res.std_err["omega"],
        lower=0,
    )
    alpha = pm.TruncatedNormal(
        "alpha",
        mu=prior_res.params["alpha[1]"],
        sigma=5 * prior_res.std_err["alpha[1]"],
        lower=0,
        upper=1,
    )
    beta = pm.TruncatedNormal(
        "beta",
        mu=prior_res.params["beta[1]"],
        sigma=5 * prior_res.std_err["beta[1]"],
        lower=0,
        upper=1,
    )
    sigma_0 = np.array(prior_demeaned.std().values, dtype=np.float64)

    # return process
    y_t = pm.GARCH11(
        "y",
        omega=omega,
        alpha_1=alpha,
        beta_1=beta,
        initial_vol=sigma_0,
        observed=data,
    )
    # prior = pm.sample_prior_predictive(15)
    # return model


# ------------------
# CREATE DAG AND SAMPLE MODEL
# ------------------
def run(name):
    model = garch11
    # # DAG
    graph = pm.model_to_graphviz(model)
    graph.render(
        f"dag_{name}", format="png", cleanup=True, directory=f"{latex_dir}Graphics",
    )
    # Sampling
    with model:
        step = pm.Metropolis()
        # trace = pm.sample(500, step=step, tune=100, chains=2)
        trace = pm.sample(
            draws=4500,
            step=step,
            chains=4,
            cores=4,
            tune=500,
            random_seed=random_state,
        )
    # Results
    output.save_to_pickle(f"{name}_model", {name: model})
    # output.save_to_pickle(f"trace_{name}", trace)


if __name__ == "__main__":
    run(name)
