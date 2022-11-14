"""
"""
import os

import pandas as pd
import numpy as np
import scipy.stats as scs
import pymc3 as pm
from tqdm import tqdm
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt

theano.config.cxx = ""
from arch import arch_model

from MyUtils.dataprocessing import DataOutputting

random_state = 13
np.random.seed(random_state)
# ==================================================
# READ IN DATA
# ==================================================
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

from A3_DataImport import train

output = DataOutputting()

latex_dir = "../Latex/"
arma_results = output.open_from_pickle("prior_arma_results")
# ==================================================
# FIRST EXERCISE DIFFERENT 'models'
# ==================================================
## Define all models
models = {}
# Plain normal returns
normal_returns = pm.Model()
with normal_returns:
    # data
    data = pm.Data("data", train)
    # priors
    mu = pm.Normal(
        "mu",
        mu=arma_results["ARMA_0_0"].params["const"],
        sigma=5 * arma_results["ARMA_0_0"].bse["const"],
    )
    sigma = pm.HalfCauchy("sigma", beta=5)

    # returns
    returns = pm.Normal("returns", mu=mu, sigma=sigma, observed=data)

models["normal_returns"] = normal_returns

# Plain Student-t returns
student_returns = pm.Model()
with student_returns:
    # data
    data = pm.Data("data", train)
    # priors
    mu = pm.Normal(
        "mu",
        mu=arma_results["ARMA_0_0"].params["const"],
        sigma=5 * arma_results["ARMA_0_0"].bse["const"],
    )
    sigma = pm.HalfCauchy("sigma", beta=5)
    nu = pm.DiscreteUniform("nu", 0, 20)
    # returns
    returns = pm.StudentT("returns", mu=mu, sigma=sigma, nu=nu, observed=data)
models["studentT_returns"] = student_returns


# AR processes
for a in range(1, 6):
    ar_model = pm.Model()
    with ar_model:
        # data
        data = pm.Data("data", train)
        # priors
        rho = []
        for i in range(a + 1):
            rho.append(
                pm.Normal(
                    f"rho_{i}",
                    mu=arma_results[f"ARMA_{a}_0"].params.iloc[i],
                    sigma=5 * arma_results[f"ARMA_{a}_0"].bse.iloc[i],
                )
            )
        sigma = pm.HalfCauchy("sigma", beta=5)

        # returns
        returns = pm.AR("returns", rho=rho, sigma=sigma, observed=data, constant=True)
    models[f"AR{a}"] = ar_model

# ARMA models
arma11 = pm.Model()
with arma11:
    # data
    data = pm.Data("data", train.T.values[0])
    # priors
    sigma = pm.HalfCauchy("sigma", beta=5)
    mu = pm.Normal(
        "mu",
        mu=arma_results[f"ARMA_1_1"].params.iloc[0],
        sigma=5 * arma_results[f"ARMA_1_1"].bse.iloc[0],
    )
    rho = pm.Normal(
        "rho",
        mu=arma_results[f"ARMA_1_1"].params.iloc[1],
        sigma=5 * arma_results[f"ARMA_1_1"].bse.iloc[1],
    )
    theta = pm.Normal(
        "theta",
        mu=arma_results[f"ARMA_1_1"].params.iloc[2],
        sigma=5 * arma_results[f"ARMA_1_1"].bse.iloc[2],
    )

    err0 = data[0] - (mu + rho * mu)

    def calc_next(last_y, this_y, err, mu, rho, theta):
        nu_t = mu + rho * last_y + theta * err
        return this_y - nu_t

    err, _ = theano.scan(
        fn=calc_next,
        sequences=dict(input=data, taps=[-1, 0]),
        outputs_info=[err0],
        non_sequences=[mu, rho, theta],
    )

    pm.Potential("like", pm.Normal.dist(0, sigma=sigma).logp(err))

# models["ARMA11"] = arma11

arma22 = pm.Model()
with arma22:
    # data
    data = pm.Data("data", train.T.values[0])
    # priors
    sigma = pm.HalfCauchy("sigma", beta=5)
    mu = pm.Normal(
        "mu",
        mu=arma_results[f"ARMA_2_2"].params.iloc[0],
        sigma=5 * arma_results[f"ARMA_2_2"].bse.iloc[0],
    )
    rho = [
        pm.Normal(
            "rho_1",
            mu=arma_results[f"ARMA_2_2"].params.iloc[1],
            sigma=5 * arma_results[f"ARMA_2_2"].bse.iloc[1],
        ),
        pm.Normal(
            "rho_2",
            mu=arma_results[f"ARMA_2_2"].params.iloc[2],
            sigma=5 * arma_results[f"ARMA_2_2"].bse.iloc[2],
        ),
    ]
    theta = [
        pm.Normal(
            "theta_1",
            mu=arma_results[f"ARMA_2_2"].params.iloc[3],
            sigma=5 * arma_results[f"ARMA_2_2"].bse.iloc[3],
        ),
        pm.Normal(
            "theta_2",
            mu=arma_results[f"ARMA_2_2"].params.iloc[4],
            sigma=5 * arma_results[f"ARMA_2_2"].bse.iloc[4],
        ),
    ]

    err0 = data[0] - (mu + rho[0] * mu + rho[1] * mu)
    err1 = data[1] - (mu + rho[0] * mu + rho[1] * mu + theta[0] * err0)

    def calc_next(y_t_2, y_t_1, y_t, err_t_2, err_t_1, mu, rho, theta):
        nu_t = (
            mu
            + rho[0] * y_t_1
            + rho[1] * y_t_2
            + theta[0] * err_t_1
            + theta[1] * err_t_2
        )
        return y_t - nu_t

    err, _ = theano.scan(
        fn=calc_next,
        sequences=dict(input=data, taps=[-2, -1, 0]),
        outputs_info=dict(initial=tt.stack([err0, err1]), taps=[-2, -1]),
        non_sequences=[mu, rho, theta],
    )

    pm.Potential("like", pm.Normal.dist(0, sigma=sigma).logp(err))

# models["ARMA22"] = arma22

arma33 = pm.Model()
with arma33:
    # data
    data = pm.Data("data", train.T.values[0])
    # priors
    sigma = pm.HalfCauchy("sigma", beta=5)
    mu = pm.Normal(
        "mu",
        mu=arma_results[f"ARMA_3_3"].params.iloc[0],
        sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[0],
    )
    rho = [
        pm.Normal(
            "rho_1",
            mu=arma_results[f"ARMA_3_3"].params.iloc[1],
            sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[1],
        ),
        pm.Normal(
            "rho_2",
            mu=arma_results[f"ARMA_3_3"].params.iloc[2],
            sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[2],
        ),
        pm.Normal(
            "rho_3",
            mu=arma_results[f"ARMA_3_3"].params.iloc[3],
            sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[3],
        ),
    ]
    theta = [
        pm.Normal(
            "theta_1",
            mu=arma_results[f"ARMA_3_3"].params.iloc[4],
            sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[4],
        ),
        pm.Normal(
            "theta_2",
            mu=arma_results[f"ARMA_3_3"].params.iloc[5],
            sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[5],
        ),
        pm.Normal(
            "theta_3",
            mu=arma_results[f"ARMA_3_3"].params.iloc[6],
            sigma=5 * arma_results[f"ARMA_3_3"].bse.iloc[6],
        ),
    ]

    err0 = data[0] - (mu + rho[0] * mu + rho[1] * mu + rho[2] * mu)
    err1 = data[1] - (mu + rho[0] * mu + rho[1] * mu + rho[2] * mu + theta[0] * err0)
    err2 = data[2] - (
        mu + rho[0] * mu + rho[1] * mu + rho[2] * mu + theta[0] * err1 + theta[1] * err0
    )

    def calc_next(y_t_3, y_t_2, y_t_1, y_t, err_t_3, err_t_2, err_t_1, mu, rho, theta):
        nu_t = (
            mu
            + rho[0] * y_t_1
            + rho[1] * y_t_2
            + rho[2] * y_t_3
            + theta[0] * err_t_1
            + theta[1] * err_t_2
            + theta[2] * err_t_3
        )
        return y_t - nu_t

    err, _ = theano.scan(
        fn=calc_next,
        sequences=dict(input=data, taps=[-3, -2, -1, 0]),
        outputs_info=dict(initial=tt.stack([err0, err1, err2]), taps=[-3, -2, -1]),
        non_sequences=[mu, rho, theta],
    )

    pm.Potential("like", pm.Normal.dist(0, sigma=sigma).logp(err))

# models["ARMA33"] = arma33


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

                # traces[model] = pm.sample(
                #     draws=4500,
                #     step=step,
                #     chains=4,
                #     cores=4,
                #     tune=500,
                #     random_seed=random_state,
                # )
                # output.save_to_pickle(f"trace_{model}", traces[model])
        except:
            # print(f"!!!!!!!!!!!!!!!!! {model} FAILED !!!!!!!!!!!!!!!!!!!!!!")
            # traces[model] = None
            continue
    # output.save_to_pickle("A5_traces_testing", traces)
    # output.save_to_pickle("A5_priors", priors)


# models = {"test": normal_returns, "test2": student_returns}
output.save_to_pickle("arma_models", models)
if __name__ == "__main__":
    # pass
    run(models)
