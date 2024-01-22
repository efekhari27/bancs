#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot
from bancs import *


omegap = "((kp / mp) ^ 0.5)"
omegas = "((ks / ms) ^ 0.5)"
gamma = "(ms / mp)"
zetaa = "((zetap + zetas) / 2)"
omegaa = f"(({omegap} + {omegas}) / 2)"
theta = f"(({omegap} - {omegas}) / {omegaa})"
Exs2 = f"(pi_ * S0 / (4 * zetas * {omegas} ^ 3) * ({zetaa} * zetas / (zetap * zetas * (4 * {zetaa}^2 + {theta}^2) + {gamma} * {zetaa}^2) * ((zetap * {omegap}^3 + zetas * {omegas}^3) * {omegap} / (4 * {zetaa} * {omegaa}^4))))"
g_string = f"Fs - ks * 3 * {Exs2}^0.5"
g = ot.SymbolicFunction(["mp", "ms", "kp", "ks", "zetap", "zetas", "Fs", "S0"], [g_string])

mp =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(1.5,     0.1))
ms =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(0.01,    0.1))
kp =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(1.,      0.2))
ks =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(0.01,    0.2))
zetap = ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(0.05,    0.4))
zetas = ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(0.02,    0.5))
Fs =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(27.5,    0.1))
S0 =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(100,     0.1))

X = ot.ComposedDistribution([mp,ms,kp,ks,zetap,zetas,Fs,S0])
X.setDescription(["$m_p$", "$m_s$", "$k_p$", "$k_s$", "$\\zeta_p$", "$\\zeta_s$", "$F_s$", "$S_0$"])
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
failure_event = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
pf_ref = 3.78 * 1e-7

N = int(5e3)
m = int(2 + np.floor(N ** (2 / (4 + 8))))

bancs = BANCS(failure_event, N=N, M=m, p0=0.25, lower_truncatures=[0.] * 8)
quantiles = bancs.run()
pf = bancs.compute_pf()
print("Quantiles =", quantiles)
print(f"Proba BANCS = {pf:.2e}")
print(f"Relative error = {np.abs(pf - pf_ref) / pf_ref:.2%}")
df = bancs.df.copy()
nb_subset = len(df['Subset'].unique())

vars_labels = list(X.getDescription()) + ["ctrl."]
multindex = pd.MultiIndex.from_product([["Subset {}".format(l + 1) for l in range(nb_subset)], vars_labels], names=["sample", "variable"])
df_tsa = pd.DataFrame([], columns=multindex, index=["HSIC", "R2HSIC", "pvalue_asymptotic", "pvalue_permutation"])
df_csa = pd.DataFrame([], columns=multindex, index=["HSIC", "R2HSIC", "pvalue_permutation"])

f = ot.SymbolicFunction(["x", "s"], ["exp(-x/s)"])
estimatorType = ot.HSICUStat()
for l in range(nb_subset):
    ssample = df[(df["Subset"]==l)][list(X.getDescription())]
    ssample["ctrl."] = np.array(ot.Normal(1).getSample(N))
    ssample = ssample.values
    ssample_output = df[(df["Subset"]==l)]["Y"].values.reshape(-1, 1)
    ssample_stds = ssample.std(axis=0)
    ssample_stds = np.append(ssample_stds, ssample_output.std())
    #ssample_stds.append(ssample_output.computeStandardDeviation()[0])
    kernel_collection = [ot.SquaredExponential([std_i]) for std_i in ssample_stds]
    dist2criticalDomain = ot.DistanceToDomainFunction(ot.Interval(quantiles[l], float("inf")))
    s = 0.1 * ssample_output.std()
    phi = ot.ParametricFunction(f, [1], [s])
    filterFunction = ot.ComposedFunction(phi, dist2criticalDomain)
    # TSA with HSIC indices
    tsa = ot.HSICEstimatorTargetSensitivity(kernel_collection, ot.Sample(ssample), ot.Sample(ssample_output), 
                                            ot.HSICUStat(), filterFunction)
    df_tsa.loc["HSIC", "Subset {}".format(l + 1)] = list(tsa.getHSICIndices())
    df_tsa.loc["R2HSIC", "Subset {}".format(l + 1)] = list(tsa.getR2HSICIndices())
    df_tsa.loc["pvalue_asymptotic", "Subset {}".format(l + 1)] = list(tsa.getPValuesAsymptotic())
    tsa.setPermutationSize(1000)
    pval = tsa.getPValuesPermutation()
    print(f"[{l}/{nb_subset}] \t T-HSIC pval = {pval}")
    df_tsa.loc["pvalue_permutation", "Subset {}".format(l + 1)] = list(pval)

for l in range(nb_subset-1):
    ssample = df[(df["Subset"]==l+1)][list(X.getDescription())]
    ssample["ctrl."] = np.array(ot.Normal(1).getSample(N))
    ssample = ssample.values
    ssample_output = df[(df["Subset"]==l+1)]["Y"].values.reshape(-1, 1)
    ssample_stds = ssample.std(axis=0)
    ssample_stds = np.append(ssample_stds, ssample_output.std())
    #ssample_stds.append(ssample_output.computeStandardDeviation()[0])
    kernel_collection = [ot.SquaredExponential([std_i]) for std_i in ssample_stds]
    dist2criticalDomain = ot.DistanceToDomainFunction(ot.Interval(quantiles[l], float("inf")))
    s = 0.1 * ssample_output.std()
    phi = ot.ParametricFunction(f, [1], [s])
    filterFunction = ot.ComposedFunction(phi, dist2criticalDomain)

    #CSA with HSIC indices
    csa = ot.HSICEstimatorConditionalSensitivity(kernel_collection, ot.Sample(ssample), ot.Sample(ssample_output), filterFunction)
    df_csa.loc["HSIC", f"Subset {l + 2}"] = list(csa.getHSICIndices())
    df_csa.loc["R2HSIC", f"Subset {l + +2}"] = list(csa.getR2HSICIndices())
    csa.setPermutationSize(1000)
    pval = csa.getPValuesPermutation()
    print(f"[{l}/{nb_subset}] \t C-HSIC pval = {pval}")
    df_csa.loc["pvalue_permutation", f"Subset {l + 2}"] = list(pval)

df_tsa_plot = df_tsa.T.reorder_levels(["variable", "sample"])
df_csa_plot = df_csa.T.reorder_levels(["variable", "sample"])
df_tsa_plot.to_csv("rosa_results/oscillator_tsa.csv")
df_csa_plot.to_csv("rosa_results/oscillator_csa.csv")
df_tsa_plot