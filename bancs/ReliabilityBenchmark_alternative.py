#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot
from bancs import BANCS
from itertools import product
from scipy.stats import bootstrap

class ReliabilityBenchmark_old:
    """
    TODO: docstring
    """
    def __init__(self, problems, methods=["BANCS", "SS", "NAIS"], sizes=[int(1e4)], nb_reps=1):
        self.nb_reps = nb_reps 
        self.sizes = sizes
        self.methods = methods
        self.problems = problems
        problem_names = [problem.getName() for problem in problems]
        self.rep_list = np.arange(nb_reps)

        multi_indexes = pd.MultiIndex.from_product([problem_names, methods, sizes, self.rep_list], names=["problem", "method", "size", "rep_idx"])
        self.df_results = pd.DataFrame([], columns=["nb_samples", "pf", "pf_ref", "pf_std"], index=multi_indexes)
        self.p0 = 0.1

    def run(self, m="amise", save_file=None):
        for problem, method, size, rep_idx in product(self.problems, self.methods, self.sizes, self.rep_list):
            size = int(size)
            dim = problem.getEvent().getAntecedent().getDimension()
            if m == "amise":
                m = int(1 + np.floor(size ** (2 / (4 + dim))))
            pf_ref = problem.getProbability()
            if method=="BANCS":
                bancs = BANCS(problem.getEvent(), N=size, M=m, p0=self.p0)
                _ = bancs.run()
                nb_samples = len(_) * size
                pf = bancs.compute_pf()
                pf_std = np.sqrt(bancs.compute_var())
            elif method=="SS":
                ss = ot.SubsetSampling(problem.getEvent())
                ss.setMaximumOuterSampling(size)
                ss.setMaximumCoefficientOfVariation(-1.0)
                ss.setConditionalProbability(self.p0)
                ss.setBlockSize(1)
                timer = ot.TimerCallback(120)
                ss.setStopCallback(timer)
                ss.run()
                res = ss.getResult()
                nb_samples = res.getOuterSampling()
                pf = res.getProbabilityEstimate()
                pf_std = res.getStandardDeviation()
            elif method=="NAIS":
                nais = ot.NAIS(problem.getEvent(), self.p0)
                nais.setMaximumOuterSampling(size)
                nais.setMaximumCoefficientOfVariation(-1.0)
                nais.setBlockSize(1)
                timer = ot.TimerCallback(120)
                nais.setStopCallback(timer)
                nais.run()
                res = nais.getResult()
                nb_samples = res.getOuterSampling()
                pf = res.getProbabilityEstimate()
                pf_std = res.getStandardDeviation()
            self.df_results.loc[(problem.getName(), method, size, rep_idx)] = [nb_samples, 
                                                                        pf, 
                                                                        pf_ref,
                                                                        pf_std,
                                                                        ]
            if rep_idx==self.nb_reps-1:
                print(f"DONE: {problem.getName()}, {method}, {size:.0E}")
            
        if save_file is not None:
            self.df_results.to_csv(save_file, index=True)