import numpy as np
import openturns as ot
import otbenchmark as otb

N = int(1e6)
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

pf_ref = 3.78 * 1e-7

#g_screened = ot.ParametricFunction(g, [7], S0.getMean())
#X = ot.ComposedDistribution([mp,ms,kp,ks,zetap,zetas,Fs])
#
#g_screened = ot.ParametricFunction(g, [6], Fs.getMean())
#X = ot.ComposedDistribution([mp,ms,kp,ks,zetap,zetas,S0])
#
g_screened = ot.ParametricFunction(g, [0], mp.getMean())
X = ot.ComposedDistribution([ms,kp,ks,zetap,zetas,Fs,S0])
#
Y = ot.CompositeRandomVector(g_screened, ot.RandomVector(X))
failure_event = ot.ThresholdEvent(Y, ot.LessOrEqual(), 0.)

# FORM IS
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(1000)
#optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)
algo = ot.FORM(optimAlgo, failure_event, X.getMean())
algo.run()
result = algo.getResult()
standardSpaceDesignPoint = result.getStandardSpaceDesignPoint()
dimension = X.getDimension()
myImportance = ot.Normal(dimension)
myImportance.setMean(standardSpaceDesignPoint)
experiment = ot.ImportanceSamplingExperiment(myImportance)
standardEvent = ot.StandardEvent(failure_event)
algo = ot.ProbabilitySimulationAlgorithm(standardEvent, experiment)
algo.setMaximumCoefficientOfVariation(-1.0)
algo.setMaximumOuterSampling(N)
algo.run()
result = algo.getResult()
pf = result.getProbabilityEstimate()

print(f"Proba = {pf:.2e}")
print(f"Relative error = {np.abs(pf - pf_ref) / pf_ref:.2%}")