#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot
from matplotlib import pyplot as plt
from bancs import *

g = ot.SymbolicFunction(['x1', 'x2', 'x3'], ['sin(x1) + 7.0 * sin(x2)^2 + 0.1 * x3^4 * sin(x1) + 10.5'])
X = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * 3)
X.setDescription(["$X_1$", "$X_2$", "$X_3$"])
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
failure_event = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
pf_ref = 1.9456000000000987e-05

N = int(5e3)
m = int(3 + np.floor(N ** (2 / (4 + 3))))

bancs = BANCS(failure_event, N=N, M=m, p0=0.25)
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
    df_tsa.loc["HSIC", f"Subset {l + 1}"] = list(tsa.getHSICIndices())
    df_tsa.loc["R2HSIC", f"Subset {l + 1}"] = list(tsa.getR2HSICIndices())
    df_tsa.loc["pvalue_asymptotic", f"Subset {l + 1}"] = list(tsa.getPValuesAsymptotic())
    tsa.setPermutationSize(1000)
    pval = list(tsa.getPValuesPermutation())
    print(f"pval perm. = {pval}")
    df_tsa.loc["pvalue_permutation", f"Subset {l + 1}"] = pval

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
    df_csa.loc["pvalue_permutation", f"Subset {l + 2}"] = list(csa.getPValuesPermutation())

df_tsa_plot = df_tsa.T.reorder_levels(["variable", "sample"])
df_csa_plot = df_csa.T.reorder_levels(["variable", "sample"])
df_tsa_plot.to_csv("rosa_results/ishigami_tsa.csv")
df_csa_plot.to_csv("rosa_results/ishigami_csa.csv")
df_tsa_plot = df_tsa_plot.reset_index()
df_csa_plot = df_csa_plot.reset_index()