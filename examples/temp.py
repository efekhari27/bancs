#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""
#%%
import numpy as np
import pandas as pd
import openturns as ot
from matplotlib import cm
from matplotlib import pyplot as plt
from bancs import *
import otbenchmark as otb
from bancs import BANCS, NAISAlgorithm, NAISResult


omegap = "((kp / mp) ^ 0.5)"
omegas = "((ks / ms) ^ 0.5)"
gamma = "(ms / mp)"
zetaa = "((zetap + zetas) / 2)"
omegaa = f"(({omegap} + {omegas}) / 2)"
theta = f"(({omegap} - {omegas}) / {omegaa})"
Exs2 = f"(pi_ * S0 / (4 * zetas * {omegas} ^ 3) * ({zetaa} * zetas / (zetap * zetas * (4 * {zetaa}^2 + {theta}^2) + {gamma} * {zetaa}^2) * ((zetap * {omegap}^3 + zetas * {omegas}^3) * {omegap} / (4 * {zetaa} * {omegaa}^4))))"
g_string = f"Fs - ks * 3 * {Exs2}^0.5"
g = ot.SymbolicFunction(["mp", "ms", "kp", "ks", "zetap", "zetas", "Fs", "S0"], [g_string])

mp =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(   1.5,     0.1))
ms =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(   0.01,    0.1))
kp =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(   1.,      0.2))
ks =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(   0.01,    0.2))
zetap = ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(0.05,    0.4))
zetas = ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(0.02,    0.5))
Fs =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(   27.5,    0.1))
S0 =    ot.ParametrizedDistribution(ot.LogNormalMuSigmaOverMu(   100,     0.1))
X = ot.ComposedDistribution([mp,ms,kp,ks,zetap,zetas,Fs,S0])

X = ot.ComposedDistribution([mp,ms,kp,ks,zetap,zetas,Fs,S0])
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
threshold = 0.
failure_event = ot.ThresholdEvent(Y, ot.LessOrEqual(), threshold)
# Reference computed using CMC with subset size N=1e7
pf_ref = 3.78 * 1e-7

#%%
N = int(1e4) 
nais = NAISAlgorithm(event=failure_event, n_IS=N, rho_quantile=0.1, lower_truncatures=[0.] * 8)
nais.run()
NAIS_result = nais.getResult()
pf = NAIS_result.getProbabilityEstimate()
print(f"Proba BANCS = {pf:.2e}")
print(f"Relative error = {np.abs(pf - pf_ref) / pf_ref:.2%}")

# %%
