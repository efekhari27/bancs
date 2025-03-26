#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot
from matplotlib import cm
from matplotlib import pyplot as plt


class BANCS:
    """
    TODO: docstring
    """
    def __init__(self, failure_event, N=int(1e4), M="AMISE", p0=0.1, lower_truncatures=[None], upper_truncatures=[None]):
        self.failure_event = failure_event
        self.N = N
        self.p0 = p0
        # Get input distribution, function, and threshold
        self.X = failure_event.getAntecedent().getDistribution()
        self.g = failure_event.getFunction()
        self.threshold = failure_event.getThreshold()
        self.xlabels = list(self.X.getDescription())
        self.dim = self.X.getDimension()
        self.fail_operator = failure_event.getOperator()
        if self.fail_operator(1, 2):
            self.quantile_order = self.p0
            self.fail_operator = np.less_equal
            self.aoperator = np.greater_equal 
        else:
            self.quantile_order = 1 - self.p0
            self.fail_operator = np.greater_equal
            self.aoperator = np.less_equal
        
        # Truncate the marginals
        if (lower_truncatures==[None]) and (lower_truncatures==[None]): 
            self.lower_truncatures = [None] * self.dim
            self.upper_truncatures = [None] * self.dim
        elif (lower_truncatures==[None]): 
            self.lower_truncatures = [None] * self.dim
            self.upper_truncatures = upper_truncatures
        elif (upper_truncatures==[None]): 
            self.lower_truncatures = lower_truncatures
            self.upper_truncatures = [None] * self.dim
        self.M = M
        self._used_bins_m = [] 
        self.df = pd.DataFrame([], columns=["Subset"] + self.xlabels + ["IS_weight", "Y", "Quantile", "Failed"])

    ## Generate a sample and compute quantile
    def generate(self, ss_index, Xcond):
        ss_indexes = np.repeat(ss_index, self.N).reshape(-1, 1)
        x_sample = np.array(Xcond.getSample(self.N))
        # Option to sample from QMC
        # sobol = ot.LowDiscrepancyExperiment(ot.SobolSequence(), Xcond, self.N)
        # sobol.setRandomize(True)
        # x_sample = sobol.generate()
        is_weight = np.array(self.X.computePDF(x_sample)) / np.array(Xcond.computePDF(x_sample))
        y_sample = np.array(self.g(x_sample))
        q_k = np.quantile(y_sample, self.quantile_order)
        q_k = np.repeat(q_k, self.N).reshape(-1, 1)
        isfailed = self.fail_operator(y_sample, q_k)
        res = np.concatenate((ss_indexes, x_sample, is_weight, y_sample, q_k, isfailed), axis=1)
        res = pd.DataFrame(res, columns=self.df.columns)
        if self.df.empty:
            return res
        else:
            return pd.concat([self.df, res])


    ## Fit a non parametric model
    def nonparametric_fit(self, ss_index):
        failed_sample = self.df[(self.df["Failed"]==1) & (self.df["Subset"]==ss_index)][self.xlabels].values
        # Including the failed samples from the previous subsets and weighting them using repetitions (inverse importance sampling mechanism)
        for k in range(ss_index): 
            failed_bool = self.fail_operator(self.df.loc[self.df["Subset"]==k, "Y"].values, self.df.loc[self.df["Subset"]==ss_index, "Quantile"].values)
            sub_df = self.df.loc[self.df["Subset"]==k]
            weighted_failed_sample = sub_df[(failed_bool)][self.xlabels].values
            nb_reps = np.floor((ss_index - k) / self.p0)
            weighted_failed_sample = np.repeat(weighted_failed_sample, nb_reps, axis=0)
            failed_sample = np.vstack([failed_sample, weighted_failed_sample])    
        marginals = []
        #Fit marginals by KDE
        #self._kde_bin_nb = len(failed_sample)
        #kernel = ot.KernelSmoothing(ot.Normal(), True, self._kde_bin_nb)
        kernel = ot.KernelSmoothing(ot.Normal(), False)
        for i in range(self.dim):
            sample = ot.Sample(failed_sample[:, i].reshape(-1, 1))
            wmix = kernel.computeMixedBandwidth(sample)
            #wmix = kernel.computeSilvermanBandwidth(sample)
            marginal = kernel.build(sample, wmix)
            if (self.lower_truncatures[i] != None) and (self.upper_truncatures[i] != None):
                marginal = ot.TruncatedDistribution(marginal, self.lower_truncatures[i], self.upper_truncatures[i])
            elif self.lower_truncatures[i] != None: 
                marginal = ot.TruncatedDistribution(marginal, self.lower_truncatures[i], ot.TruncatedDistribution.LOWER)
            elif self.upper_truncatures[i] != None: 
                marginal = ot.TruncatedDistribution(marginal, self.upper_truncatures[i], ot.TruncatedDistribution.UPPER)
            marginals.append(marginal)
        # Fit copula by EBC
        if self.M == "AMISE":
            m_opt = ot.BernsteinCopulaFactory.ComputeAMISEBinNumber(failed_sample)
        elif self.M == "Beta": 
            m_opt = np.unique(failed_sample, axis=0).shape[0]
        elif self.M == "LogLikelihood":
            m_opt = optimize_ebc_loglikehood(failed_sample, kfolds=2)
        elif self.M == "PenalizedKL":
            m_opt = ot.BernsteinCopulaFactory.ComputePenalizedCsiszarDivergenceBinNumber(np.unique(failed_sample, axis=0), ot.SymbolicFunction(['t'], ['t * ln(t)']))
        elif isinstance(self.M, int):
            m_opt = self.M
        bernstein_copula = ot.EmpiricalBernsteinCopula(failed_sample, m_opt)
        # print(f"{self.M} : {m_ebc}")
        self._used_bins_m.append(m_opt)
        return ot.ComposedDistribution(marginals, bernstein_copula)

    ## Run the algorithm
    def run(self):
        # Initial reliability problem
        ss_index = 0
        self.df = self.generate(ss_index, self.X)
        self.conditional_distributions = [self.X]
        while (self.aoperator(self.df.iloc[-1]["Quantile"], self.threshold)) and (ss_index < 20):
            #TODO raise warning on the ss_index<20 condition and add attribute to 20
            Xcond = self.nonparametric_fit(ss_index)
            self.conditional_distributions.append(Xcond)
            self.df = self.generate(ss_index + 1, Xcond)
            ss_index += 1
        quantiles = self.df["Quantile"].unique()
        self.nb_steps = len(quantiles)
        # Setup the last subset sample to failed if below the threshold
        self.df.loc[(self.fail_operator(self.df["Y"], self.threshold)) & (self.df["Subset"]==(self.nb_steps - 1)), "Failed"] = 1
        # Generally failed: samples below the threshold among all the subsets 
        self.df['gFailed'] = 0
        self.df.loc[(self.fail_operator(self.df["Y"], self.threshold)), 'gFailed'] = 1
        return quantiles
    
    def compute_pf(self):
        quantiles = self.df["Quantile"].unique()
        # BANCS estimator 1 (type Subset Sampling)
        #pf = (self.p0 ** (nb_steps - 1)) * (self.df[(self.fail_operator(self.df["Y"], self.threshold)) & (self.df["Subset"]==nb_steps - 1)].shape[0]) / self.N
        # BANCS estimator 2 (type Nonadaptive importance sampling keeping all the failed samples)
        #pf = self.df.loc[self.df['gFailed']==1, "IS_weight"].sum() / (nb_steps * self.N)
        # BANCS estimator 3 (type Nonadaptive importance sampling)
        pf = self.df.loc[(self.df["Subset"]==self.nb_steps - 1) & (self.df['Failed']==1), "IS_weight"].sum() / self.N        
        return pf

    def compute_var(self):
        #var_pf = (np.mean(self.df.loc[self.df['gFailed']==1, "IS_weight"] ** 2) - self.compute_pf() ** 2) / (self.N-1)
        var_pf = (np.mean(self.df.loc[(self.df["Subset"]==self.nb_steps - 1) & (self.df['Failed']==1), "IS_weight"] ** 2) - self.compute_pf() ** 2) / (self.N-1)
        return var_pf
    
    def draw_2D_BANCS(self, title="", colorbar=cm.Greys_r):
        quantiles = self.df["Quantile"].unique()
        nb_steps = len(quantiles)
        d = DrawFunctions()
        #d.set_bounds([-2] * 2, [8] * 2)
        fig = d.draw_2D_controur(title, function=self.g, distribution=self.X, colorbar=colorbar, contour_values=False)
        for i in range(nb_steps):
            failed_sample = self.df[(self.df["Subset"]==i) & (self.df["Failed"]==1)]
            x0 = failed_sample[self.xlabels[0]].values
            x1 = failed_sample[self.xlabels[1]].values
            if i == (nb_steps - 1) : 
                quantile = 0.
            else: 
                quantile = quantiles[i]
            sslabel = fr"Sample {i+1} ($\hat{{q}}_{{[{i+1}]}}^{{p_0}} = {quantile:.3}$)"
            plt.scatter(x0, x1, color='C{}'.format(i), marker='.', alpha=0.5, label=sslabel)
            contour = plt.contour(d.X0, d.X1, d.Z, levels=[quantile], colors='C{}'.format(i), linewidths=2, linestyles=['solid'])
            plt.clabel(contour, inline=True, fontsize=12, colors='k')
        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=2, columnspacing=1.2)
        return fig


def cross_validation_loglikelihood(m, sample, kfolds=2):
    cv_loglikelihood = 0.0
    splitter = ot.KFoldSplitter(sample.getSize(), kfolds)
    for indices_train, indices_test in splitter:
        sample_train, sample_test = sample[indices_train], sample[indices_test]
        sample_train = (sample_train.rank() + 0.5) / sample_train.getSize()
        sample_test = (sample_test.rank() + 0.5) / sample_test.getSize()
        ebc = ot.EmpiricalBernsteinCopula(sample_train, int(m), True)
        cv_loglikelihood -= ebc.computeLogPDF(sample_test).computeMean()[0]
    cv_ll = cv_loglikelihood / kfolds
    # print(f"{m} : {cv_ll:.3e}")
    return cv_ll

def hill_climbing(func, start_x, bounds, jump=1, max_iter=100):
    lbound, ubound = bounds
    x = start_x
    fx = func(x)
    for _ in range(max_iter):
        left, right = max(x - jump, lbound), min(x + jump, ubound)
        f_left, f_right = func(left), func(right)
        if f_left < fx:
            x, fx = left, f_left
        elif f_right < fx:
            x, fx = right, f_right
        else:
            break
    return x, fx

def optimize_ebc_loglikehood(sample, m_min=None, m_max=None, kfolds=2):
    # Drop duplicates for the optimization
    sample = ot.Sample(sample).sortUnique()
    if m_min is None: 
        m_min = 1
    if m_max is None: 
        m_max = sample.sortUnique().getSize() 
    def func(m):
        return cross_validation_loglikelihood(m, sample=sample, kfolds=kfolds)
    m_start = ot.BernsteinCopulaFactory.ComputeAMISEBinNumber(sample)
    optimal_m, _ = hill_climbing(func, start_x=m_min, bounds=[m_min, m_max], jump=100)
    optimal_m, _ = hill_climbing(func, start_x=optimal_m, bounds=[m_min, m_max], jump=10)
    optimal_m, _ = hill_climbing(func, start_x=optimal_m, bounds=[m_min, m_max], jump=2)
    # print(f"HC optimal : {optimal_m}")
    return optimal_m

####################
class DrawFunctions:
    """
    TODO: docstring
    """
    def __init__(self):
        self.dim = 2
        self.grid_size = 500
        lowerbound = [-7.5] * self.dim
        upperbound = [7.5] * self.dim
        self.set_bounds(lowerbound, upperbound)

    def set_bounds(self, lowerbound, upperbound):
        mesher = ot.IntervalMesher([self.grid_size-1] * self.dim)
        interval = ot.Interval(lowerbound, upperbound)
        mesh = mesher.build(interval)
        self.nodes = mesh.getVertices()
        self.X0, self.X1 = np.array(self.nodes).T.reshape(self.dim, self.grid_size, self.grid_size)
        return None

    def draw_2D_controur(self, title, function=None, distribution=None, colorbar=cm.coolwarm, nb_isocurves=8, contour_values=True):
        fig = plt.figure(figsize=(7, 6))
        if distribution is not None:
            Zpdf = np.array(distribution.computePDF(self.nodes)).reshape(self.grid_size, self.grid_size)
            contours = plt.contour(self.X0, self.X1, Zpdf, nb_isocurves, colors='black', alpha=0.3)
            if contour_values:
                plt.clabel(contours, inline=True, fontsize=8)
        if function is not None:
            self.Z = np.array(function(self.nodes)).reshape(self.grid_size, self.grid_size)
            plt.contourf(self.X0, self.X1, self.Z, 10, cmap=colorbar)
            plt.colorbar()
        plt.title(title, fontsize=20)
        plt.xlabel("$x_1$", fontsize=20)
        plt.ylabel("$x_2$", fontsize=20)
        
        #plt.close()
        return fig