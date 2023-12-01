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
#ot.ResourceMap.SetAsUnsignedInteger('KernelSmoothing-BinNumber', 1000)

class BANCS:
    """
    TODO: docstring
    """
    def __init__(self, failure_event, N=int(1e4), M=None, p0=0.1, lower_truncatures=[None], upper_truncatures=[None]):
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
        if self.fail_operator(1,2):
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
        if M is None:
            ## IMSE minimization manuscript M.Lasserre (2022) p.117
            self.M = 1 + int((self.p0 * self.N) ** (2 / (self.dim + 4)))
        else:
            self.M = M
        self.df = pd.DataFrame([], columns=["Subset"] + self.xlabels + ["IS_weight", "Y", "Quantile", "Failed"])

    ## Generate a sample and compute quantile
    def generate(self, ss_index, Xcond):
        ss_indexes = np.repeat(ss_index, self.N).reshape(-1, 1)
        x_sample = np.array(Xcond.getSample(self.N))
        is_weight = np.array(self.X.computePDF(x_sample)) / np.array(Xcond.computePDF(x_sample))
        # Option to sample from QMC
        #sobol = ot.LowDiscrepancyExperiment(ot.SobolSequence(), Xcond, self.N)
        #sobol.setRandomize(True)
        #x_sample = sobol.generate()
        y_sample = np.array(self.g(x_sample))
        q_k = np.quantile(y_sample, self.quantile_order)
        q_k = np.repeat(q_k, self.N).reshape(-1, 1)
        isfailed = self.fail_operator(y_sample, q_k)
        res = np.concatenate((ss_indexes, x_sample, is_weight, y_sample, q_k, isfailed), axis=1)
        res = pd.DataFrame(res, columns=self.df.columns)
        return pd.concat([self.df, pd.DataFrame(res)])

    ## Fit a non parametric model
    def nonparametric_fit(self, ss_index):
        failed_sample = self.df[(self.df["Failed"]==1) & (self.df["Subset"]==ss_index)][self.xlabels].values
        # Including the failed samples from the previous subsets and weighting tem by repetitions
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
            marginal = kernel.build(sample, wmix)
            if (self.lower_truncatures[i] != None) and (self.upper_truncatures[i] != None):
                marginal = ot.TruncatedDistribution(marginal, self.lower_truncatures[i], self.upper_truncatures[i])
            elif self.lower_truncatures[i] != None: 
                marginal = ot.TruncatedDistribution(marginal, self.lower_truncatures[i], ot.TruncatedDistribution.LOWER)
            elif self.upper_truncatures[i] != None: 
                marginal = ot.TruncatedDistribution(marginal, self.upper_truncatures[i], ot.TruncatedDistribution.UPPER)
            marginals.append(marginal)
        # Fit copula by EBC
        bernstein_copula = ot.EmpiricalBernsteinCopula(failed_sample, self.M)
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
        nb_steps = len(quantiles)
        # Setup the last subset sample to failed if below the threshold
        self.df.loc[(self.fail_operator(self.df["Y"], self.threshold)) & (self.df["Subset"]==(nb_steps - 1)), "Failed"] = 1
        # Generally failed: samples below the threshold among all the subsets 
        self.df['gFailed'] = 0
        self.df.loc[(self.fail_operator(self.df["Y"], self.threshold)), 'gFailed'] = 1
        return quantiles
    
    def compute_pf(self):
        quantiles = self.df["Quantile"].unique()
        nb_steps = len(quantiles)
        # BANCS estimator 1 (type Subset Sampling)
        #pf = (self.p0 ** (nb_steps - 1)) * (self.df[(self.fail_operator(self.df["Y"], self.threshold)) & (self.df["Subset"]==nb_steps - 1)].shape[0]) / self.N
        # BANCS estimator 2 (type Nonadaptive importance sampling)
        pf = self.df.loc[self.df['gFailed']==1, "IS_weight"].sum() / (nb_steps * self.N)
        return pf

    def compute_var(self):
        var_pf = (np.mean(self.df.loc[self.df['gFailed']==1, "IS_weight"] ** 2) - self.compute_pf() ** 2) / (self.N-1)
        return var_pf
    
    def draw_2D_BANCS(self, title="", colorbar=cm.Greys_r):
        quantiles = self.df["Quantile"].unique()
        nb_steps = len(quantiles)
        d = DrawFunctions()
        #d.set_bounds([-4] * 2, [10] * 2)
        fig = d.draw_2D_controur(title, function=self.g, distribution=self.X, colorbar=colorbar, contour_values=False)
        for i in range(nb_steps):
            failed_sample = self.df[(self.df["Subset"]==i) & (self.df["Failed"]==1)]
            x0 = failed_sample[self.xlabels[0]].values
            x1 = failed_sample[self.xlabels[1]].values
            if i == (nb_steps - 1) : 
                quantile = 0.
            else: 
                quantile = quantiles[i]
            sslabel = fr"Subset {i+1} ($\hat{{q}}_{{[{i+1}]}}^\alpha = {quantile:.3}$)"
            plt.scatter(x0, x1, color='C{}'.format(i), marker='.', alpha=0.5, label=sslabel)
            contour = plt.contour(d.X0, d.X1, d.Z, levels=[quantile], colors='C{}'.format(i), linewidths=2, linestyles=['solid'])
            plt.clabel(contour, inline=True, fontsize=12, colors='k')
        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=2, columnspacing=1.2)
        return fig


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