import numpy as np
import openturns as ot
import otbenchmark as otb

N = int(1e4)
g = ot.SymbolicFunction(['x1', 'x2', 'x3'], ['sin(x1) + 7.0 * sin(x2)^2 + 0.1 * x3^4 * sin(x1) + 10.5'])
pf_ref = 1.9456000000000987e-05

g_screened = ot.ParametricFunction(g, [2], [0.])
X = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * 2)
Y = ot.CompositeRandomVector(g_screened, ot.RandomVector(X))
failure_event = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)

ss = ot.SubsetSampling(failure_event)
ss.setMaximumOuterSampling(N)
ss.setMaximumCoefficientOfVariation(-1.0)
ss.setConditionalProbability(0.3)
ss.setBlockSize(10)
ss.run()
res = ss.getResult()
nb_samples = res.getOuterSampling()
pf = res.getProbabilityEstimate()
print(f"Proba = {pf:.2e}")
print(f"Relative error = {np.abs(pf - pf_ref) / pf_ref:.2%}")