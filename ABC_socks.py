"""
copy of (in python instead of R) + additions
https://www.youtube.com/watch?v=nKCT-Cdk0xY
http://www.sumsar.net/blog/2014/10/tiny-data-and-the-socks-of-karl-broman/
"""
import os

import pandas as pd
import numpy as np
import scipy.stats as scs
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
from collections import Counter
from MyUtils.dataprocessing import DataOutputting
import time
import datetime

random_state = 13
np.random.seed(random_state)
os.chdir(f"{os.getenv('BAYESIAN_PROJECT')}/Python")
latex_dir = "../Latex/"
# ==================================================
# create data generating process
# ==================================================

"""
For a given number of 'paired' socks and 'odd' socks (paired*2+odds=total),
put them in washing machine and randomlydraw 'n' socks
"""
n_pairs = 9
n_odds = 5
picks = 11


def pick_socks(n_pairs, n_odds, picks):
    """Data generating process, for a given number of 'paired'
    socks and 'odd' socks (paired*2+odds=total), put them in washing machine and randomly draw 'n' socks

    Args:
        n_pairs (int): number of pairs (=number of paired socks/2)
        n_odds (int): number of odds
        picks (int): number of random draws

    Returns:
        tuple: (number of pairs in randow drawn series, number of odds, proportion of pairs)
    """
    # create array of socks, each sock is represented by a number,
    # if the sock comes in a pair, both socks have the same number
    socks = np.concatenate(
        [np.repeat(np.arange(n_pairs), 2), np.arange(n_pairs, n_pairs + n_odds)]
    )

    # after an imagenary tumble  in the washing maching, take out socks randomly
    # in case there is an inconsistency in inputs
    if len(socks) == picks:
        return n_pairs, n_odds
    assert len(socks) >= picks, "'picks' is larger than total number of socks"

    # picks socks randomly
    picked_socks = pd.Series(np.sort(np.random.choice(socks, picks, replace=False)))
    #  count pairs
    sock_counts = picked_socks.value_counts()
    uniques = len(sock_counts[sock_counts == 1])
    pairs = len(sock_counts[sock_counts == 2])

    return pairs, uniques


# test function
pick_socks(n_pairs, n_odds, picks)

# ==================================================
# PRIORS
# ==================================================
# _ PRIOR ON TOTAL NUMBER OF SOCKS = variable of interest
# easier to think about mean and sd
prior_average_n_socks = 30
prior_sd_n_socks = 15
# however, python takes n and p as inputs for NegativeBinomial
prior_p = prior_average_n_socks / prior_sd_n_socks ** 2
prior_n = prior_p * prior_average_n_socks / (1 - prior_p)

prior_number_of_socks = scs.nbinom(n=prior_n, p=prior_p)
prior_number_of_socks.stats()
# prior_number_of_socks = scs.poisson(mu=prior_average_n_socks)


# PRIOR ON PROPORTION n_pairs*2/total (because inputs for DGP is n_pairs and n_odds)
# again specify mean and standard deviation
prior_proportion_mean = 0.95
prior_proportion_sd = 0.10

# compute a and b parameter for beta distribution
from scipy.optimize import fsolve

# goes a bit faster than if I have to do it manually :)
def f(a):
    b = a * (1 - prior_proportion_mean) / prior_proportion_mean
    variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
    return variance - prior_proportion_sd ** 2


a = fsolve(f, 0.5)[0]
b = a * (1 - prior_proportion_mean) / prior_proportion_mean

prior_proportion = scs.beta(a=a, b=b)

# plot prior to see if it makes sense
# plot priors
fig, ax = plt.subplots(1, 2)
x = list(range(100))
y = prior_number_of_socks.pmf(x)
ax[0].bar(x, y)
ax[0].set_title("Number of socks")
x = np.linspace(0, 1, 51)
y = prior_proportion.pdf(x)
ax[1].plot(x, y)
ax[1].set_title("Proportion of pairs")
fig.savefig(f"{latex_dir}Graphics/socks_priors.png")
fig.show()

# ==================================================
# ABS algorithms
# ==================================================
# define observed data
picks = 11
observed_pairs = 0
observed_odds = 11
N_draws = 10000

# ----------------------------
# EXACT REJECTION SAMPLING ABC
# ----------------------------
print("EXACT REJECTION SAMPLING ABC")
# Now here I will deviate form the blogpost code, because the author simply runs
# his simulation N times and then intersect on an exact match, this causes  the number of  actual
# draws to be 'random', so I'm implementing the one from our slides and force the draws equaling N
# by simple rejection
def exact_rejection_ABC(N):
    draws = pd.DataFrame(
        index=["n_socks", "proportion", "pairs", "odds"], columns=range(N)
    )
    counter = 0
    for n in tqdm(range(N)):
        accept = False
        while not accept:  # keeps on drawing until
            counter += 1
            # first draw from 'number of socks' prior,
            # BUT I keep drawing until  this number is at least equal to picks
            simulated_total_socks = 0
            while simulated_total_socks < picks:
                simulated_total_socks = prior_number_of_socks.rvs()

            # draw from proportion prior
            simulated_proportion = prior_proportion.rvs()
            # compute  inputs for DGP: pairs and odds
            simulated_pairs = int(simulated_total_socks * simulated_proportion / 2)
            simulated_odds = simulated_total_socks - 2 * simulated_pairs
            # DGP: simulate a random draw  from waching machine given inputs
            simulated_drawn_pairs, simulated_drawn_odds = pick_socks(
                simulated_pairs, simulated_odds, picks
            )
            # check equality to observations
            # (note that equality of one implies the other as well, but I explicitely  write the two)
            if (simulated_drawn_pairs == observed_pairs) & (
                simulated_odds == observed_odds
            ):
                accept = True

        draws[n] = [
            simulated_total_socks,
            simulated_proportion,
            simulated_drawn_pairs,
            simulated_drawn_odds,
        ]
    draws = draws.T
    acceptance_rate = N / counter
    return draws, acceptance_rate


start = time.time()
sub, acceptance_rate = exact_rejection_ABC(N_draws)
end = time.time()
total_time = datetime.timedelta(seconds=np.round(end - start, 0))

# plot posteriors
fig, ax = plt.subplots(2, 2, figsize=(18, 12), sharex="col", sharey="row")
sns.distplot(sub["n_socks"], ax=ax[0][0])
ax[0][0].set_title("Exact rejection")
print(
    f"Estimated total number of socks = {int(sub['n_socks'].mean())} "
    f"with AR of {np.round(acceptance_rate*100, 2)}% "
    f"and took {str(total_time)}"
)
print()
# ----------------------------
# BASIC REJECTION SAMPLING ABC
# ----------------------------
print("BASIC REJECTION SAMPLING ABC")
# Acceptance is very low,  so to speed things up, I will try the Basic ABC where the simulated date
# must be 'close enough'.
#  I will take as distance simply the euclidean distance between the observed vector  [pairs,odds]
# and the simulated version
def euclidean_distance(a, b):
    if isinstance(a, int):
        a, b = [a], [b]
    if len(a) != len(b):
        raise ValueError("lengths  of vector do not match")

    return np.sqrt(np.array([(a[i] - b[i]) ** 2 for i in range(len(a))]).sum())


def basis_rejection_ABC(N, epsilon, distance=euclidean_distance):
    draws = pd.DataFrame(
        index=["n_socks", "proportion", "pairs", "odds"], columns=range(N)
    )
    counter = 0
    for n in tqdm(range(N)):
        accept = False
        while not accept:  # keeps on drawing until
            counter += 1
            # first draw from 'number of socks' prior,
            # BUT I keep drawing until  this number is at least equal to picks
            simulated_total_socks = 0
            while simulated_total_socks < picks:
                simulated_total_socks = prior_number_of_socks.rvs()

            # draw from proportion prior
            simulated_proportion = prior_proportion.rvs()
            # compute  inputs for DGP: pairs and odds
            simulated_pairs = int(simulated_total_socks * simulated_proportion / 2)
            simulated_odds = simulated_total_socks - 2 * simulated_pairs
            # DGP: simulate a random draw  from waching machine given inputs
            simulated_drawn_pairs, simulated_drawn_odds = pick_socks(
                simulated_pairs, simulated_odds, picks
            )
            # check equality to observations
            # (note that equality of one implies the other as well, but I explicitely  write the two)
            if (
                distance(
                    [simulated_drawn_pairs, simulated_drawn_odds],
                    [observed_pairs, observed_odds],
                )
                <= epsilon
            ):
                accept = True

        draws[n] = [
            simulated_total_socks,
            simulated_proportion,
            simulated_drawn_pairs,
            simulated_drawn_odds,
        ]
    draws = draws.T
    acceptance_rate = N / counter
    return draws, acceptance_rate


# with an epsilon of 2.5 I allow for the two values to be within a distance of 2, but not together
start = time.time()
sub, acceptance_rate = basis_rejection_ABC(
    N_draws, epsilon=1, distance=euclidean_distance
)
end = time.time()
total_time = datetime.timedelta(seconds=np.round(end - start, 0))

sns.distplot(sub["n_socks"], ax=ax[0][1])
ax[0][1].set_title("Basic rejection")
fig.show()
print(
    f"Estimated total number of socks = {int(sub['n_socks'].mean())} "
    f"with AR of {np.round(acceptance_rate*100, 2)}% "
    f"and took {str(total_time)}"
)
print()
# ----------------------------
# WEIGHTED SAMPLING ABC
# ----------------------------
print("WEIGHTED SAMPLING ABC")


def Epanechnikov(u):
    return 3 / 4 * (1 - u ** 2)


def max_normalisation(x):
    maximum = x.max()
    if maximum == 0:
        return x
    else:
        return x / maximum


def weighted_rejection_ABC(
    N, distance=euclidean_distance, normalizer=max_normalisation, kernel=Epanechnikov
):
    draws = pd.DataFrame(
        index=["n_socks", "proportion", "pairs", "odds", "distance"], columns=range(N)
    )
    for n in tqdm(range(N)):
        # first draw from 'number of socks' prior,
        # BUT I keep drawing until  this number is at least equal to picks
        simulated_total_socks = 0
        while simulated_total_socks < picks:
            simulated_total_socks = prior_number_of_socks.rvs()

        # draw from proportion prior
        simulated_proportion = prior_proportion.rvs()
        # compute  inputs for DGP: pairs and odds
        simulated_pairs = int(simulated_total_socks * simulated_proportion / 2)
        simulated_odds = simulated_total_socks - 2 * simulated_pairs
        # DGP: simulate a random draw  from waching machine given inputs
        simulated_drawn_pairs, simulated_drawn_odds = pick_socks(
            simulated_pairs, simulated_odds, picks
        )

        draws[n] = [
            simulated_total_socks,
            simulated_proportion,
            simulated_drawn_pairs,
            simulated_drawn_odds,
            distance(
                [simulated_drawn_pairs, simulated_drawn_odds],
                [observed_pairs, observed_odds],
            ),
        ]
    draws = draws.T
    draws["distance_normalized"] = normalizer(draws["distance"])
    draws["weights"] = kernel(draws["distance_normalized"])
    draws["weights_normalized"] = draws["weights"] / draws["weights"].sum()
    return draws


# with an epsilon of 2.5 I allow for the two values to be within a distance of 2, but not together
start = time.time()
sub = weighted_rejection_ABC(
    N_draws,
    distance=euclidean_distance,
    normalizer=max_normalisation,
    kernel=Epanechnikov,
)
end = time.time()
total_time = datetime.timedelta(seconds=np.round(end - start, 0))


kernel = scs.gaussian_kde(sub["n_socks"], weights=sub["weights"])
x = list(range(100))
y = kernel(x)
ax[1][0].plot(x, y)
ax[1][0].set_xlabel("n_socks")
ax[1][0].set_title("Weighted  ABC")


print(
    f"Estimated total number of socks = {int((sub['n_socks'] * sub['weights_normalized']).sum())} "
    f"and took {str(total_time)}"
)
print()

# ----------------------------
# k-NN SAMPLING ABC
# ----------------------------
print("k-NN SAMPLING ABC")
k = int(N_draws / 10)


def kNN_rejection_ABC(
    N, k, distance=euclidean_distance, normalizer=max_normalisation, kernel=Epanechnikov
):
    # I still want N accepted draws, so I'm gonna rescale the inputted k and N
    share = k / N
    new_k = N
    new_N = int(N / share)

    draws = pd.DataFrame(
        index=["n_socks", "proportion", "pairs", "odds", "distance"],
        columns=range(new_N),
    )

    for n in tqdm(range(new_N)):
        # first draw from 'number of socks' prior,
        # BUT I keep drawing until  this number is at least equal to picks
        simulated_total_socks = 0
        while simulated_total_socks < picks:
            simulated_total_socks = prior_number_of_socks.rvs()

        # draw from proportion prior
        simulated_proportion = prior_proportion.rvs()
        # compute  inputs for DGP: pairs and odds
        simulated_pairs = int(simulated_total_socks * simulated_proportion / 2)
        simulated_odds = simulated_total_socks - 2 * simulated_pairs
        # DGP: simulate a random draw  from waching machine given inputs
        simulated_drawn_pairs, simulated_drawn_odds = pick_socks(
            simulated_pairs, simulated_odds, picks
        )

        draws[n] = [
            simulated_total_socks,
            simulated_proportion,
            simulated_drawn_pairs,
            simulated_drawn_odds,
            distance(
                [simulated_drawn_pairs, simulated_drawn_odds],
                [observed_pairs, observed_odds],
            ),
        ]
    draws = draws.T

    # sort and select
    draws = draws.sort_values("distance").reset_index(drop=True)
    draws = draws.iloc[:new_k]

    draws["distance_normalized"] = normalizer(draws["distance"])
    draws["weights"] = kernel(draws["distance_normalized"])
    draws["weights_normalized"] = draws["weights"] / draws["weights"].sum()

    return draws


start = time.time()
sub = kNN_rejection_ABC(
    N=N_draws,
    k=k,
    distance=euclidean_distance,
    normalizer=max_normalisation,
    kernel=Epanechnikov,
)
end = time.time()
total_time = datetime.timedelta(seconds=np.round(end - start, 0))


kernel = scs.gaussian_kde(sub["n_socks"], weights=sub["weights"])
x = list(range(100))
y = kernel(x)
ax[1][1].plot(x, y)
ax[1][1].set_title("k-NN ABC")
ax[1][1].set_xlabel("n_socks")
fig.savefig(f"{latex_dir}Graphics/socks_posteriors.png")


print(
    f"Estimated total number of socks = {int((sub['n_socks'] * sub['weights_normalized']).sum())} "
    f"and took {str(total_time)}"
)

