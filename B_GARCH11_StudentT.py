"""
BASED ON CH11 in 'Bayesian methhods in Finance
"""
import os

import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as scs
import theano
import theano.tensor as tt
from arch import arch_model
from pymc3.distributions import continuous, distribution
from theano import scan


from MyUtils.dataprocessing import DataOutputting

theano.config.cxx = ""


random_state = 13
np.random.seed(random_state)
# ==================================================
# READ IN DATA
# ==================================================
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

from A3_DataImport import prior_demeaned, train_demeaned
from B0_ML_GARCH import garch11_ml_results as prior_res

output = DataOutputting()

latex_dir = "../Latex/"

# ==================================================
# WRITING OWN CLASS/DISTRIBUTION
# ==================================================


class GARCH11_StudentT(distribution.Continuous):
    def __init__(self, omega, alpha_1, beta_1, initial_vol, nu, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.omega = omega = tt.as_tensor_variable(omega)
        self.alpha_1 = alpha_1 = tt.as_tensor_variable(alpha_1)
        self.beta_1 = beta_1 = tt.as_tensor_variable(beta_1)
        self.initial_vol = tt.as_tensor_variable(initial_vol)
        self.nu = nu = tt.as_tensor_variable(nu)
        self.mean = tt.as_tensor_variable(0.0)

    def get_volatility(self, x):
        x = x[:-1]

        def volatility_update(x, vol, w, a, b):
            return tt.sqrt(w + a * tt.square(x) + b * tt.square(vol))

        vol, _ = scan(
            fn=volatility_update,
            sequences=[x],
            outputs_info=[self.initial_vol],
            non_sequences=[self.omega, self.alpha_1, self.beta_1],
        )
        return tt.concatenate([[self.initial_vol], vol])

    def logp(self, x):
        """
        Calculate log-probability of GARCH(1, 1) distribution at specified value.
        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.
        Returns
        -------
        TensorVariable
        """
        vol = self.get_volatility(x)
        return tt.sum(continuous.StudentT.dist(nu=self.nu, mu=0.0, sigma=vol).logp(x))


# ==================================================
# CREATE MODEL: GARCH(1, 1)
# ==================================================
# general
name = "garch11_studentT"

# ------------------
# TRAINING
# ------------------
# model specification
def create_model():
    model = pm.Model()
    with model:
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

        nu = pm.DiscreteUniform("nu", 0, 20)

        sigma_0 = np.array(prior_demeaned.std().values, dtype=np.float64)

        # return process
        y_t = GARCH11_StudentT(
            "y",
            omega=omega,
            alpha_1=alpha,
            beta_1=beta,
            initial_vol=sigma_0,
            nu=nu,
            observed=data,
        )
        # prior = pm.sample_prior_predictive(15)
    return model


# ------------------
# CREATE DAG AND SAMPLE MODEL
# ------------------
def run(name):
    model = create_model()
    # DAG
    graph = pm.model_to_graphviz(model)
    graph.render(
        f"dag_{name}", format="png", cleanup=True, directory=f"{latex_dir}Graphics",
    )
    # Sampling
    with model:
        step = pm.Metropolis()
        # trace = pm.sample(50, step=step, tune=10, chains=2)
        trace = pm.sample(
            draws=4500,
            step=step,
            chains=4,
            cores=4,
            tune=500,
            random_seed=random_state,
        )
    # Results
    output.save_to_pickle(f"trace_{name}", trace)
    output.save_to_pickle(f"{name}_model", {name: model})


if __name__ == "__main__":
    run(name)
