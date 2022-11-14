"""
"""
import os

# import pandas as pd
import numpy as np

# import scipy.stats as scs
import pymc3 as pm
from tqdm import tqdm
import pandas as pd
import scipy.stats as scs

# from pymc3.distributions import continuous
# from pymc3.distributions import continuous, distribution
# import theano.tensor as tt
# from theano import scan

# from arch import arch_model
import theano

theano.config.cxx = ""
from MyUtils.dataprocessing import DataOutputting

import matplotlib.pyplot as plt
import seaborn as sns

random_state = 13
np.random.seed(random_state)
# ==================================================
# READ IN DATA
# ==================================================
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")

from A3_DataImport import test, test_demeaned, train_demeaned, prior_demeaned
from B_GARCH11_StudentT import GARCH11_StudentT

from B0_ML_GARCH import garch11_ml_results

output = DataOutputting()

latex_dir = "../Latex/"

arma_results = output.open_from_pickle("prior_arma_results")
# ==================================================
# READ IN MODELSs
# ==================================================
arma_models = output.open_from_pickle("arma_models")
sv_models = output.open_from_pickle("sv_models")
garch11_model = output.open_from_pickle("garch11_model")
garch11student_model = output.open_from_pickle("garch11_studentT_model")
garch_models = garch11_model
garch_models.update(garch11student_model)

models = {}
models.update(arma_models)
models.update(garch_models)
models.update(sv_models)
# # models
# models = [
#     # "test",
#     # "test2",
#     # # "normal_returns",
#     # # "studentT_returns",
#     # # "AR1",
#     # # "AR2",
#     # # "AR3",
#     # "ARMA11",
#     # "ARMA22",
#     # "ARMA33",
#     # "garch11",
#     "garch11_studentT",
# ]
# models = models + list(sv_models.keys())
# ==================================================
# READ IN TRACES
# ==================================================
# getting the summaries and traces
summaries = {}
traces = {}
for name in tqdm(models, desc="Sumaries loop"):
    # read  in
    traces[name] = output.open_from_pickle(f"trace_{name}")
    # if name not in sv_models:
    #     try:
    #         summaries[name] = output.read_csv(f"trace_summary_{name}")
    #     except FileNotFoundError:
    #         # get summary
    #         trace_summary = pm.summary(traces[name])
    #         output.to_csv(trace_summary, f"trace_summary_{name}")
    #         summaries[name] = trace_summary
# traces[name]['volatility']
# ==================================================
# MAKE POSTERIOR PLOTS
# ==================================================
for name in tqdm(models, desc="Posterior plots"):
    trace = traces[name]
    varnames = [x for x in trace.varnames if "__" not in x]

    for varname in varnames:
        # chain histograms
        fig, ax = plt.subplots()
        for i in range(trace.nchains):
            t = trace.get_values(varname=varname, chains=i)
            sns.distplot(t, hist=False, ax=ax, norm_hist=True, label=i + 1)
            ax.legend()
            ax.set_xlabel(varname)
        fig.savefig(f"{latex_dir}Graphics/{name}_{varname}_posterior_chains.png")
        plt.close()

        fig, ax = plt.subplots()
        sns.distplot(trace[varname], hist=False, ax=ax, norm_hist=True)
        ax.set_xlabel(varname)
        fig.savefig(f"{latex_dir}Graphics/{name}_{varname}_posterior.png")
        plt.close()


for m in arma_results:
    print(m)
    print(arma_results[m].params)
    print(arma_results[m].bse)
    print(5 * arma_results[m].bse)
    print()

for m in arma_models:
    print(m)
    for x in [x for x in traces[m].varnames if "__" not in x]:

        print(
            f"{x}: {np.round(traces[m][x].mean(),3)} - {np.round(np.percentile(traces[m][x], [2.5, 97.5]),3)}"
        )
    print()

u = np.linspace(0.01, 0.99, 100)

# normal_return - mu
fig, ax = plt.subplots()

prior = scs.norm(loc=-0.087, scale=0.252)
x = prior.ppf(u)
y = prior.pdf(x)
sns.lineplot(x, y, label="$p(\mu)$", ax=ax)

sns.distplot(traces["normal_returns"]["mu"], ax=ax, label="$p(\mu|r)$")

ax.set_xlabel("$\mu$")
ax.legend()
fig.savefig(f"{latex_dir}Graphics/normal_returns_mu_prior_posterior.png")
fig.show()

# student_t_return - mu
fig, ax = plt.subplots()

prior = scs.norm(loc=-0.087, scale=0.252)
x = prior.ppf(u)
y = prior.pdf(x)
sns.lineplot(x, y, label="$p(\mu)$", ax=ax)

sns.distplot(traces["studentT_returns"]["mu"], ax=ax, label="$p(\mu|r)$")
ax.set_xlabel("$\mu$")
ax.legend()
fig.savefig(f"{latex_dir}Graphics/studentT_returns_mu_prior_posterior.png")
fig.show()
# ==================================================
# PREDICT and evaluate TIME SERIES MODELS
# ==================================================
def rmse(pred, test):
    error = test - pred
    error2 = error ** 2
    root_mean_squared_error = np.sqrt(error2.mean()).values[0]
    return (error, root_mean_squared_error)


# name = list(models.keys())[0]
# name = "AR2"
# name = "normal_returns"


predictions = {}
predictions_mean = {}
errors = pd.DataFrame(index=test.index)
rmsds = pd.DataFrame(index=["RMSE"])

for name in tqdm(models):
    model = models[name]
    trace = traces[name]

    try:
        predictions[name] = output.open_from_pickle(f"p_{name}")
        predictions_mean[name] = output.open_from_pickle(f"pred_mean_{name}")
    except FileNotFoundError:
        if name in ["normal_returns", "studentT_returns"]:
            # with model:
            #     pm.set_data({"data": test})
            #     preds = pm.sample_posterior_predictive(trace)
            # predictions[name] = pd.DataFrame(preds["returns"].T[0], index=test.index)
            preds = np.array([(traces[name]["mu"])])

            preds = np.repeat(preds, len(test), axis=0)
            predictions[name] = pd.DataFrame(preds, index=test.index)
        elif name in ["AR1", "AR2", "AR3", "AR4", "AR5"]:
            r = int(name[-1])
            n = len(trace["rho_0"])
            preds = np.zeros((len(test), n))
            preds[0:r] = test[:r]
            rho = np.array([trace[f"rho_{r}"] for r in range(r + 1)]).T
            sigma = trace["sigma"]
            # i = 0
            for i in tqdm(range(n)):
                # if (rho[i][1:].sum() > 1):
                #     preds[:, i] = np.nan
                # eps = sigma[i] * np.random.randn(len(test))
                # t = r
                for t in range(r, len(test)):
                    preds[t, i] = rho[i][0] + rho[i][-r:] @ np.flip(
                        test[t - r : t].T.values[0]
                    )
                    # preds[t, i] = (
                    #     rho[i][0] + rho[i][-r:] @ np.flip(preds[t - r : t, i]) + eps[t]
                    # )
            predictions[name] = pd.DataFrame(preds, index=test.index)

        elif name in [f"ARMA{r}{r}" for r in range(1, 4)]:

            r = int(name[-1])
            if r != 1:
                continue
            n = len(trace["mu"])
            preds = np.zeros((len(test), n))
            preds[0:r] = test[:r]
            eps = np.zeros((len(test), n))

            with model:
                pm.set_data({"data": test.iloc[:2].values.flatten()})
                f = pm.sample_posterior_predictive(trace=trace)

            if r == 1:
                rho = trace["rho"]
                theta = trace["theta"]
            else:
                rho = np.array([trace[f"rho_{r}"] for r in range(1, r + 1)]).T
                theta = np.array([trace[f"theta_{r}"] for r in range(1, r + 1)]).T

            sigma = trace["sigma"]
            mu = trace["mu"]
            # i = 4739
            for i in tqdm(range(n)):
                if np.abs(rho[i].sum()) + np.abs(theta[i].sum()) > 2:
                    preds[:, i] = np.nan
                    eps[:, i] = np.nan
                else:
                    # t = r
                    for t in range(r, len(test)):
                        if r == 1:
                            preds[t, i] = (
                                mu[[i]]
                                + rho[i] * preds[t - r : t, i]
                                + theta[i] * eps[t - r : t, i]
                            )
                        else:
                            preds[t, i] = (
                                mu[[i]]
                                + rho[i] @ np.flip(preds[t - r : t, i])
                                + theta[i] @ np.flip(eps[t - r : t, i])
                            )
                        eps[t, i] = test.iloc[t].values[0] - preds[t, i]
            predictions[name] = pd.DataFrame(preds, index=test.index)
        else:
            continue

    predictions_mean[name] = pd.DataFrame(
        predictions[name].mean(axis=1), columns=test.columns
    )
    output.save_to_pickle(f"p_{name}", predictions[name])
    output.save_to_pickle(f"pred_mean_{name}", predictions_mean[name])

    error2, rmsd = rmse(predictions_mean[name], test)
    errors[name] = error2
    rmsds[name] = rmsd


print(rmsds.T)
fig, ax = plt.subplots(2, 1, figsize=(20, 12))
test[5:250].plot(ax=ax[0], label="Bel20")
test[250:].plot(ax=ax[1], label="Bel20")
for m in models:
    try:
        series = predictions_mean[m].copy()
        series.columns = [m]
        series[5:250].plot(ax=ax[0], label=m)
        series[250:].plot(ax=ax[1])

    except KeyError:
        continue
ax[0].legend(ncol=2)
ax[0].set_title("(a) First half")
ax[1].set_title("(b) Second half")
fig.savefig(f"{latex_dir}Graphics/timeseries_models_predictions.png")
fig.show()

fig, ax = plt.subplots(2, 1, figsize=(20, 12))
for m in models:
    try:
        errors[m][5:250].plot(ax=ax[0], label=m)
        errors[m][250:].plot(ax=ax[1])
    except KeyError:
        continue
ax[0].legend(ncol=2)
ax[0].set_title("(a) First half")
ax[1].set_title("(b) Second half")
fig.savefig(f"{latex_dir}Graphics/timeseries_models_errors.png")
fig.show()


fig, ax = plt.subplots(2, 1, figsize=(20, 12))
test[5:250].plot(ax=ax[0], label="Bel20")
test[250:].plot(ax=ax[1])
for m in models:
    try:
        predictions[m][5:].quantile(0.05, axis=1)[5:250].plot(ax=ax[0], label=m)
        predictions[m][5:].quantile(0.05, axis=1)[250:].plot(ax=ax[1])

    except KeyError:
        continue
ax[0].legend(ncol=2)
ax[0].set_title("(a) First half")
ax[1].set_title("(b) Second half")
fig.savefig(f"{latex_dir}Graphics/timeseries_var.png")
fig.show()

# ==================================================
# PREDICT and evaluate VOLATILITY MODELS
# ==================================================
# garch models
sigma2_t = {}
for name in garch_models:
    n = len(traces[name]["alpha"])
    sigma2 = np.zeros((len(train_demeaned) + 1, n))
    sigma2[0:1] = prior_demeaned.var()

    y_t = np.concatenate([prior_demeaned[-1:], train_demeaned]).T[0]
    # i = 0
    for i in tqdm(range(n)):
        omega = traces[name]["omega"][i]
        alpha = traces[name]["alpha"][i]
        beta = traces[name]["beta"][i]
        if alpha + beta > 1:
            sigma2[:, i] = np.nan
        # t= 1
        for t in range(1, len(train_demeaned) + 1):
            sigma2[t, i] = omega + alpha * y_t[t - 1] ** 2 + beta * sigma2[t - 1, i]

    sigma2_t[name] = pd.DataFrame(sigma2[1:, :], index=train_demeaned.index)

fig, ax = plt.subplots(2, 1, sharex=True)
for name in garch_models:
    ax[0].plot(np.sqrt(sigma2_t[name].mean(axis=1)), label=name)
ax[1].plot(train_demeaned)

ax[0].legend()
ax[1].legend()
fig.show()


for m in garch_models:
    print(m)
    for x in [x for x in traces[m].varnames if "__" not in x]:

        print(
            f"{x}: {np.round(traces[m][x].mean(),3)} - {np.round(np.percentile(traces[m][x], [2.5, 97.5]),3)}"
        )
    print()

print(garch11_ml_results)

# sv_models
vols = {}
for name in sv_models:
    model = models[name]
    trace = traces[name]
    vols[name] = pd.DataFrame(trace["volatility_process"].T, index=train_demeaned.index)


for m in sv_models:
    print(m)
    for x in [x for x in traces[m].varnames if "__" not in x]:

        print(
            f"{x}: {np.round(traces[m][x].mean(),3)} - {np.round(np.percentile(traces[m][x], [2.5, 97.5]),3)}"
        )
    print()


# PLOT ALL
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16, 24))
for name in sv_models:
    ax[1].plot(vols[name].mean(axis=1), label=name)
for name in garch_models:
    ax[0].plot(np.sqrt(sigma2_t[name].mean(axis=1)), label=name)
ax[2].plot(train_demeaned)

ax[0].legend()
ax[0].set_title("(a) GARCH models")


ax[1].legend()
ax[1].set_title("(b) SV models")

ax[2].set_title("(c) BEL20 returns")


fig.savefig(f"{latex_dir}Graphics/sv_models_plots.png")
fig.show()


# volatility of volatility prior posteriors plot
fig, ax = plt.subplots(1, 2, figsize=(15, 10))

u = np.linspace(0.01, 0.99, 100)
prior = scs.halfcauchy(loc=-0, scale=5)
x = prior.ppf(u)
y = prior.pdf(x)

sns.lineplot(x, y, label="$p(\\tau)$", ax=ax[0])
sns.lineplot(x, y, label="$p(\\tau)$", ax=ax[1])

sns.distplot(
    traces["SV_ar1_normal"]["s"],
    ax=ax[0],
    label="$p(\\tau|r)$",
    norm_hist=True,
    # hist=False,
)
ax[0].set_xlabel("$\\tau$")
ax[0].legend()
ax[0].set_title("(a) Posterior and prior of $\\tau$")
ax[1].set_xlabel("$\\tau$")
ax[1].legend()
ax[1].set_title("(b) (Uninformative) prior for $\\tau$")
ax[0].set_xlim((0, 0.5))
fig.savefig(f"{latex_dir}Graphics/tau_prior_posterior.png")

fig.show()

