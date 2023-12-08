#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""
from bancs import ReliabilityBenchmark
import openturns as ot
import otbenchmark as otb

# 
problem4B = otb.FourBranchSerialSystemReliability()
g1 = "3 + 0.1 * (x1 - x2) ^ 2 - (x1 + x2) / sqrt(2)"
g2 = "3 + 0.1 * (x1 - x2) ^ 2 + (x1 + x2) / sqrt(2)"
g3 = "(x1 - x2) + 7 / sqrt(2)"
g4 = "(x2 - x1) + 7 / sqrt(2)"
g = ot.SymbolicFunction(["x1", "x2"], ["min({}, {}, {}, {})".format(g1, g2, g3, g4)])
X = ot.ComposedDistribution([ot.Normal(0., 1.)] * 2)
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
problem4B.thresholdEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
problem4B.probability = 0.0022197899999999047
#
problem_parabolic = otb.ReliabilityProblem57()
g = ot.SymbolicFunction(["x1", "x2"], ["(x1 - x2) ^ 2 - 8 * (x1 + x2 - 5)"])
X = ot.ComposedDistribution([ot.Normal(0., 1.)] * 2)
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
problem_parabolic.thresholdEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
problem_parabolic.probability = 0.0001315399999999369
problem_parabolic.name = "Parabolic"

#
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
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
failure_event = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
oscillator_problem = otb.ReliabilityProblem57()
oscillator_problem.thresholdEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
oscillator_problem.probability = 3.78 * 1e-7
oscillator_problem.name = "Oscillator"

# Sinusoidal 2D, not so rare
problemRP53 = otb.ReliabilityProblem53()
# Almost linear but 7D
problemRP38 = otb.ReliabilityProblem38()
problem4B.probability = 0.008043179999996148
#
ebc_tuning = "amise"
nb_reps = 100
bench_sizes = [300, 500, 700, int(1e3), int(2e3), int(5e3), int(1e4)]

## RP parabolic ##
#####################
rb = ReliabilityBenchmark(problems=[problem_parabolic], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes)
rb.run(m="amise", save_file=f"./bancs_results/Parabolic_results_{ebc_tuning}.csv")

## RP 53 ##
#####################
#rb = ReliabilityBenchmark(problems=[problemRP53], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes)
#rb.run(m=ebc_tuning, save_file=f"./bancs_results/RP53_results_{ebc_tuning}.csv")

## RP 38 ##
#####################
#rb = ReliabilityBenchmark(problems=[problemRP38], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes)
#rb.run(reps=nb_reps, m=ebc_tuning, save_file=f"./bancs_results/RP38_results_{ebc_tuning}.csv")

## RP four-branch ##
#####################
#rb = ReliabilityBenchmark(problems=[problem4B], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes)
#rb.run(reps=nb_reps, m=ebc_tuning, save_file=f"./bancs_results/RP4B_results_{ebc_tuning}.csv")

## Oscillator ##
################
#rb = ReliabilityBenchmark(problems=[oscillator_problem], methods=["BANCS", "SS"], sizes=bench_sizes)
#rb.run(reps=nb_reps, m=ebc_tuning, save_file=f"./bancs_results/Oscillator_results_{ebc_tuning}.csv")