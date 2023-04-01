#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""
from bancs import ReliabilityBenchmark
import openturns as ot
import otbenchmark as otb

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

problemRP57 = otb.ReliabilityProblem57()
equations = ["var g1 := -x1^2 + x2^3 + 3"]
equations.append("var g2 := 2 - x1 - 8 * x2")
equations.append("var g3 := (x1 + 3)^2 + (x2 + 3)^2 - 4")
equations.append("gsys := min(max(g1, g2), g3) ")
formula = ";".join(equations)
g = ot.SymbolicFunction(["x1", "x2"], ["gsys"], formula)
X = ot.ComposedDistribution([ot.Normal(0., 0.6)] * 2)
Y = ot.CompositeRandomVector(g, ot.RandomVector(X))
problemRP57.thresholdEvent = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)
problemRP57.probability = 0.0009812000000000679

problemRP38 = otb.ReliabilityProblem38()

## RP four-branch ##
# AMISE tuning
rb = ReliabilityBenchmark(problems=[problem4B], methods=["BANCS", "SS", "NAIS"], sizes=[int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)])
ebc_tuning = "amise"
rb.run(reps=100, m=ebc_tuning, save_file=f"RP4B_results_{ebc_tuning}.csv")

# Beta tuning 
rb = ReliabilityBenchmark(problems=[problem4B], methods=["BANCS"], sizes=[int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)])
ebc_tuning = "beta"
rb.run(reps=100, m=ebc_tuning, save_file=f"RP4B_results_{ebc_tuning}.csv")

## RP 57 ##
# AMISE tuning
rb = ReliabilityBenchmark(problems=[problemRP57], methods=["BANCS", "SS", "NAIS"], sizes=[int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)])
ebc_tuning = "amise"
rb.run(reps=100, m=ebc_tuning, save_file=f"RP57_results_{ebc_tuning}.csv")
# Beta tuning
rb = ReliabilityBenchmark(problems=[problemRP57], methods=["BANCS"], sizes=[int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)])
ebc_tuning = "beta"
rb.run(reps=100, m=ebc_tuning, save_file=f"RP57_results_{ebc_tuning}.csv")

## RP 38 ##
# AMISE tuning
rb = ReliabilityBenchmark(problems=[problemRP38], methods=["BANCS", "SS", "NAIS"], sizes=[int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)])
ebc_tuning = "amise"
rb.run(reps=100, m=ebc_tuning, save_file=f"RP38_results_{ebc_tuning}.csv")
# Beta tuning 
rb = ReliabilityBenchmark(problems=[problemRP38], methods=["BANCS"], sizes=[int(1e3), int(2e3), int(4e3), int(6e3), int(8e3), int(1e4)])
ebc_tuning = "beta"
rb.run(reps=100, m=ebc_tuning, save_file=f"RP38_results_{ebc_tuning}.csv")
