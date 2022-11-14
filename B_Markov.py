"""
"""
import os

import pandas as pd
import numpy as np
import scipy.stats as scs
import pymc3 as pm
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

from A3_DataImport import train_demeaned, prior_demeaned, index_samples

# from B0_ML_GARCH import garch11_ml_results as prior_res

output = DataOutputting()

latex_dir = "../Latex/"

# ==================================================
# TEST 1
# ==================================================
k = 3
ndata = 500
spread = 5
centers = np.array([-spread, 0, spread])

# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] + np.random.randn(ndata)

# setup model
model = pm.Model()
with model:
    # cluster sizes
    p = pm.Dirichlet("p", a=np.array([1.0, 1.0, 1.0]), shape=k)
    # ensure all clusters have some points
    p_min_potential = pm.Potential(
        "p_min_potential", tt.switch(tt.min(p) < 0.1, -np.inf, 0)
    )

    # cluster centers
    means = pm.Normal("means", mu=[0, 0, 0], sigma=15, shape=k)
    # break symmetry
    order_means_potential = pm.Potential(
        "order_means_potential",
        tt.switch(means[1] - means[0] < 0, -np.inf, 0)
        + tt.switch(means[2] - means[1] < 0, -np.inf, 0),
    )

    # measurement error
    sd = pm.Uniform("sd", lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical("category", p=p, shape=ndata)

    # likelihood for each observed value
    points = pm.Normal("obs", mu=means[category], sigma=sd, observed=data)

# fit model
with model:
    step1 = pm.Metropolis(vars=[p, sd, means])
    step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2])
    tr = pm.sample(10000, step=[step1, step2], tune=5000)

pm.traceplot(tr)
plt.show()

# ==================================================
# TEST 2
# http://modernstatisticalworkflow.blogspot.com/2018/02/regime-switching-models-in-stan.html
# https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/master/Ports/Regime-switching%20models%20in%20PyMC3.ipynb
# ==================================================
import theano.tensor as tt
import theano

returns = theano.shared(index_samples["train"])

theano.config.test_values = "raise"
theano.config.print_test_value = True
## TESTING THE RECURRENT Function
#  set eta = distribution of data, given state
eta_ = tt.dmatrix("eta_")
eta0 = np.random.rand(100, 2)
eta_.tag.test_value = eta0

# test for a P - transition matrix
P = tt.dmatrix("P")
P0 = np.asarray([[0.75, 0.25], [0.25, 0.75]])
P.tag.test_value = P0

# set xi - probability of being in state j at time t
xi_ = tt.dscalar("xi_")
xi_.tag.test_value = 0.75

# dont really get this yet
xi_out = tt.dscalar("xi_out")
xi_out.tag.test_value = 0
ft_out = tt.dscalar("ft_out")
ft_out.tag.test_value = 0


def ft_xit_dt(Eta, ft, Xi, P):
    Xi_ = tt.shape_padleft(Xi)
    xit0 = tt.stack([Xi_, 1 - Xi_], axis=1).T
    ft = tt.sum(tt.dot(xit0 * P, Eta))
    Xi1 = (P[0, 0] * Xi + P[1, 0] * (1 - Xi)) * Eta[0] / ft
    return [ft, Xi1]


([ft, xi], updates) = theano.scan(
    ft_xit_dt, sequences=eta_, outputs_info=[ft_out, xi_out], non_sequences=P
)

ft_xit_dt_ = theano.function(
    inputs=[eta_, ft_out, xi_out, P], outputs=[ft, xi], updates=updates
)


ft1, xi1 = ft_xit_dt_(eta0, 0, 0.75, P0)

# next few things are basically testing if the defined function above works as expected
ft2 = np.zeros(100)
xi2 = np.zeros(100)

ftfunc = (
    lambda eta, xi: P0[0, 0] * xi * eta[0]
    + P0[0, 1] * xi * eta[1]
    + P0[1, 1] * (1 - xi) * eta[1]
    + P0[1, 0] * (1 - xi) * eta[0]
)
Eta = eta0[0]
Xi_ = np.asarray([0.75])
ft2[0] = ftfunc(Eta, Xi_)
xi2[0] = (P0[0, 0] * Xi_ + P0[1, 0] * (1 - Xi_)) * Eta[0] / ft2[0]


for i in range(1, 100):
    Eta = eta0[i]
    Xi_ = xi2[i - 1]
    ft2[i] = ftfunc(Eta, Xi_)
    xi2[i] = (P0[0, 0] * Xi_ + P0[1, 0] * (1 - Xi_)) * Eta[0] / ft2[i]


np.testing.assert_almost_equal(ft1, ft2)
np.testing.assert_almost_equal(xi1, xi2)


with pm.Model() as m:
    # Transition matrix
    p = pm.Beta("p", alpha=10.0, beta=2.0, shape=2)
    P = tt.diag(p)
    P = tt.set_subtensor(P[0, 1], 1 - p[0])
    P = tt.set_subtensor(P[1, 0], 1 - p[1])

    # eta
    alpha = pm.Normal("alpha", mu=0.0, sd=0.1, shape=2)
    sigma = pm.HalfCauchy("sigma", beta=1.0, shape=2)
    eta1 = tt.exp(pm.Normal.dist(mu=alpha[0], sd=sigma[0]).logp(yshared))

    y_tm1_init = pm.Normal("y_init", mu=0.0, sd=0.1)
    pNormal = pm.Bound(pm.Normal, lower=0.0)
    rho = pNormal("rho", mu=1.0, sd=0.1, testval=1.0)
    eta2 = tt.zeros_like(eta1)
    eta2 = tt.set_subtensor(
        eta2[0],
        tt.exp(
            pm.Normal.dist(mu=alpha[1] + rho * y_tm1_init, sd=sigma[1]).logp(yshared[0])
        ),
    )
    eta2 = tt.set_subtensor(
        eta2[1:],
        tt.exp(
            pm.Normal.dist(mu=alpha[1] + rho * yshared[:-1], sd=sigma[1]).logp(
                yshared[1:]
            )
        ),
    )

    eta = tt.stack([eta1, eta2], axis=1)

    xi_init = pm.Beta("xi_init", alpha=2.0, beta=2.0)
    ft_out = theano.shared(0.0)  # place holder
    ([ft, xi], updates) = theano.scan(
        ft_xit_dt, sequences=eta, outputs_info=[ft_out, xi_init], non_sequences=P
    )

    Xi = pm.Deterministic("Xi", xi)
    # likelihood `target += sum(log(f))`
    pm.Potential("likelihood", tt.sum(tt.log(ft)))

