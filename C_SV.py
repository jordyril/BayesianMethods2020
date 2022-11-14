"""
SOURCES/INSPIRATION:
- CH12 in 'Bayesian methhods in Finance
- https://docs.pymc.io/notebooks/stochastic_volatility.html
- kim1998stochastic
- jacquier2002bayesian
"""
import os
from pprint import pprint

import pandas as pd
import numpy as np
import pymc3 as pm
from tqdm import tqdm

from MyUtils.dataprocessing import DataOutputting

np.random.seed(13)
# -------------------------------------
# READ IN DATA
# -------------------------------------
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

from A3_DataImport import train

output = DataOutputting()

latex_dir = "../Latex/"
arma_results = output.open_from_pickle("prior_arma_results")
random_state = 13
# -------------------------------------
# BASIC SV - OBSERVED
# -------------------------------------
# define models
models = {}
# GRW
sv_grw_normal = pm.Model()
with sv_grw_normal:
    # data
    data = pm.Data("data", train.T.values[0])

    # priors
    mu = pm.Normal(
        "mu",
        mu=arma_results["ARMA_0_0"].params["const"],
        sigma=5 * arma_results["ARMA_0_0"].bse["const"],
    )
    s = pm.HalfCauchy("s", beta=5)

    # volatility
    h = pm.GaussianRandomWalk("h", sigma=s, shape=len(train))
    volatility_process = pm.Deterministic("volatility_process", pm.math.exp(h / 2))

    returns = pm.Normal("returns", mu=mu, sigma=volatility_process, observed=data)

models["SV_GRW_normal"] = sv_grw_normal

# GRW student T
sv_grw_student = pm.Model()
with sv_grw_student:
    # data
    data = pm.Data("data", train.T.values[0])

    # priors
    nu = pm.DiscreteUniform("nu", 0, 20)
    mu = pm.Normal(
        "mu",
        mu=arma_results["ARMA_0_0"].params["const"],
        sigma=5 * arma_results["ARMA_0_0"].bse["const"],
    )
    s = pm.HalfCauchy("s", beta=5)

    # volatility
    h = pm.GaussianRandomWalk("h", sigma=s, shape=len(train))
    volatility_process = pm.Deterministic("volatility_process", pm.math.exp(h / 2))

    returns = pm.StudentT(
        "returns", mu=mu, nu=nu, sigma=volatility_process, observed=data
    )

models["SV_GRW_student"] = sv_grw_student

# AR(1)
sv_ar1_normal = pm.Model()
with sv_ar1_normal:
    # data
    data = pm.Data("data", train.T.values[0])

    # priors
    mu = pm.Normal(
        "mu",
        mu=arma_results["ARMA_0_0"].params["const"],
        sigma=5 * arma_results["ARMA_0_0"].bse["const"],
    )
    rho = [pm.HalfCauchy("omega", beta=5), pm.Uniform("rho", -1, 1)]
    s = pm.HalfCauchy("s", beta=5)
    # volatility
    h = pm.AR("h", rho=rho, sigma=s, shape=len(train), constant=True)
    volatility_process = pm.Deterministic("volatility_process", pm.math.exp(h / 2))
    returns = pm.Normal("returns", mu=mu, sigma=volatility_process, observed=data)

models["SV_ar1_normal"] = sv_ar1_normal

# AR(1) - Student-t
sv_ar1_student = pm.Model()
with sv_ar1_student:
    # data
    data = pm.Data("data", train.T.values[0])

    # priors
    mu = pm.Normal(
        "mu",
        mu=arma_results["ARMA_0_0"].params["const"],
        sigma=5 * arma_results["ARMA_0_0"].bse["const"],
    )
    # rho = pm.Uniform("rho", -1, 1)
    rho = [pm.HalfCauchy("omega", beta=5), pm.Uniform("rho", -1, 1)]
    s = pm.HalfCauchy("s", beta=5)
    nu = pm.DiscreteUniform("nu", 0, 20)
    # volatility
    h = pm.AR("h", rho=rho, sigma=s, shape=len(train), constant=True)
    volatility_process = pm.Deterministic("volatility_process", pm.math.exp(h / 2))

    returns = pm.StudentT(
        "returns", mu=mu, nu=nu, sigma=volatility_process, observed=data
    )

models["SV_ar1_student"] = sv_ar1_student


# sample/estimate models
def run(models):
    traces = {}
    for model in tqdm(models, desc="Model loop"):

        try:
            with models[model]:
                # DAG
                graph = pm.model_to_graphviz(models[model])
                graph.render(
                    f"dag_{model}",
                    format="png",
                    cleanup=True,
                    directory=f"{latex_dir}Graphics",
                )
                # step = pm.Metropolis()
                # traces[model] = pm.sample(
                #     draws=500, step=step, tune=10, random_seed=random_state, chains=2
                # )

                traces[model] = pm.sample(
                    draws=4500,
                    # step=step,
                    chains=4,
                    cores=4,
                    tune=500,
                    random_seed=random_state,
                )
                output.save_to_pickle(f"trace_{model}", traces[model])
        except:
            print(f"!!!!!!!!!!!!!!!!! {model} FAILED !!!!!!!!!!!!!!!!!!!!!!")
            traces[model] = None
            continue


# models = {"test": normal_returns, "test2": student_returns}
output.save_to_pickle("sv_models", models)
if __name__ == "__main__":
    # pass
    run(models)
