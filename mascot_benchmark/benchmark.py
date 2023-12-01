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
problem4B.probability = 0.0022130999999990827
#
problem_parabolic = otb.ReliabilityProblem57()
g = ot.SymbolicFunction(["x1", "x2"], ["(x1 - x2) ^ 2 - 8 * (x1 + x2 - 5)"])
X = ot.ComposedDistribution([ot.Normal(0., 1.)] * 2)
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
problem_parabolic.thresholdEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
problem_parabolic.probability = 0.0001315399999999369
problem_parabolic.name = "Parabolic"
#
problemRP53 = otb.ReliabilityProblem53()
#
problemRP38 = otb.ReliabilityProblem38()

ebc_tuning = "amise"
nb_reps = 100
bench_sizes = [300, 500, 700, int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)]
## RP parabolic ##
#####################
# AMISE tuning
#rb = ReliabilityBenchmark(problems=[problem_parabolic], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes, nb_reps=nb_reps)
#rb.run(m=ebc_tuning, save_file=f"./results/Parabolic_results_{ebc_tuning}.csv")

## RP four-branch ##
#####################
# AMISE tuning
rb = ReliabilityBenchmark(problems=[problem4B], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes, nb_reps=nb_reps)
rb.run(m=ebc_tuning, save_file=f"./results/RP4B_results_{ebc_tuning}.csv")

## RP 53 ##
#####################
# AMISE tuning
#rb = ReliabilityBenchmark(problems=[problemRP53], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes, nb_reps=nb_reps)
#rb.run(m=ebc_tuning, save_file=f"./results/RP53_results_{ebc_tuning}.csv")

## RP 38 ##
#####################
# AMISE tuning
#rb = ReliabilityBenchmark(problems=[problemRP38], methods=["BANCS", "SS", "NAIS"], sizes=bench_sizes, nb_reps=nb_reps)
#rb.run(m=ebc_tuning, save_file=f"./results/RP38_results_{ebc_tuning}.csv")
