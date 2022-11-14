"""
"""
import os

import pandas as pd

# import numpy as np

from MyUtils.dataprocessing import DataProcessor
from pythonlatex import Figure, Table
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# -------------------------------------
# READ IN DATA + OPTIONS
# -------------------------------------
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")


from A3_DataImport import index, index_returns, index_samples

figure = Figure(folders_path="../Latex/", position="H")
table = Table(folders_path="../Latex/", position="H")
start = index.index[0].date().strftime("%d/%m/%Y")
end = index.index[-1].date().strftime("%d/%m/%Y")
# -------------------------------------
# EXPLORATION
# -------------------------------------
# index + return plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(index)
ax1.set_ylabel("Price")
ax1.set_title("(a) Level")
ax1.set_xlabel("Date")
ax2.plot(index_returns)
ax2.set_xlabel("Date")
ax2.set_ylabel("Log-returns")
ax2.set_title("(b) Returns")
plot_acf(index_returns, ax=ax3, zero=False, title="(c) Autocorrelation", lags=50)
ax3.set_xlabel("Lag")
ax3.set_ylabel("AC")
fig.subplots_adjust(hspace=1)

figure.create_input_latex(
    "bel20_level_returns",
    caption="Bel20 price level and return series",
    description=f"""Figure presents time-series for the Bel20 index. Panel (a) shows the price level, while the series of daily returns is plotted in panel (b). Panel (c) contains the autocorrelations up to a lag of 50 days. The sample period ranges from {start} to {end}.""",
    tight_layout=True,
)
figure.reset(show=False, close=True)

# histograms
fig, ax = plt.subplots()
sns.distplot(index_returns, fit=stats.norm, ax=ax)
ax.legend([f"Normal", "Actual"])

figure.create_input_latex(
    "bel20_hist",
    caption="Histogram of Bel20 returns",
    description=f"""Figure presents the histogram for the Bel20 index returns and the fit of a Normal distribution. The sample period ranges from {start} to {end}""",
    tight_layout=True,
)
figure.reset(show=False, close=True)


# Descriptive stats table
descriptives = pd.DataFrame(index=["Mean", "Stdv", "Skew", "Ex. kurt",])

for data, name in zip(
    [index_returns] + [index_samples[key] for key in index_samples.keys()],
    ["full"] + list(index_samples.keys()),
):
    temp_l = []
    for mom in ["mean", "std", "skew", "kurt"]:
        temp_l.append(getattr(data, mom)().values[0])

    descriptives[name] = temp_l

descriptives.columns = ["Full", "Prior", "Train", "Test"]

table.create_input_latex(
    descriptives.T,
    "bel20_descriptives",
    caption="Summary statistics Bel20 returns",
    description=f"""Table presents the average (Mean), standard deviation (Stdv), skewness (Skew) and excess kurtosis (Es. kurt) of Bel20 returns for different subsets. The full sample (Full) ranges from {start} to {end}, while Prior, Train, Test containing each 15%, 75% and 10% of the full sample respectively.""",
    adjustbox=False,
    float_format="{:0.2f}".format,
)
