import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm


from A3_DataImport import train

np.random.seed(0)

returns = pd.read_csv(pm.get_data("SP500.csv"), index_col="Date")
returns["change"] = np.log(returns["Close"]).diff()
returns = returns.dropna()
returns.head()

fig, ax = plt.subplots(figsize=(14, 4))
returns.plot(y="change", label="S&P 500", ax=ax)
ax.set(xlabel="time", ylabel="returns")
ax.legend()
fig.show()

sv_GRW_student = pm.Model()
with sv_GRW_student:
    step_size = pm.Exponential("step_size", 10)
    volatility = pm.GaussianRandomWalk(
        "volatility", sigma=step_size, shape=len(returns["change"])
    )
    nu = pm.Exponential("nu", 0.1)
    ret = pm.StudentT(
        "returns", nu=nu, lam=np.exp(-2 * volatility), observed=returns["change"]
    )

graph = pm.model_to_graphviz(sv_GRW_student)
graph.view()

with sv_GRW_student:
    prior = pm.sample_prior_predictive(500)

# fig, ax = plt.subplots(figsize=(14, 4))
# returns["change"].plot(ax=ax, lw=1, color="black")
# ax.plot(prior["returns"][4:6].T, "g", alpha=0.5, lw=1, zorder=-10)
# max_observed, max_simulated = (
#     np.max(np.abs(returns["change"])),
#     np.max(np.abs(prior["returns"])),
# )
# ax.set_title(
#     f"Maximum observed: {max_observed:.2g}\nMaximum simulated: {max_simulated:.2g}(!)"
# )
# fig.show()

with sv_GRW_student:
    step = pm.Metropolis()
    trace = pm.sample(200, tune=50, step=step)

with sv_GRW_student:
    posterior_predictive = pm.sample_posterior_predictive(trace)

returns["change"]
pd.DataFrame(posterior_predictive["returns"]).T.mean(axis=1)
returns
# pm.traceplot(trace)
plt.show()

fig, ax = plt.subplots(figsize=(14, 4))


y_vals = np.exp(trace["volatility"])[::5].T
x_vals = np.vstack([returns.index for _ in y_vals.T]).T.astype(np.datetime64)

plt.plot(x_vals, y_vals, "k", alpha=0.002)
ax.set_xlim(x_vals.min(), x_vals.max())
ax.set_ylim(bottom=0)
ax.set(title="Estimated volatility over time", xlabel="Date", ylabel="Volatility")
fig.show()

fig, axes = plt.subplots(nrows=2, figsize=(14, 7), sharex=True)
returns["change"].plot(ax=axes[0], color="black")

axes[1].plot(np.exp(trace["volatility"][::100].T), "r", alpha=0.5)
axes[0].plot(posterior_predictive["returns"][::100].T, "g", alpha=0.5, zorder=-10)
axes[0].set_title(
    "True log returns (black) and posterior predictive log returns (green)"
)
axes[1].set_title("Posterior volatility")
fig.show()
