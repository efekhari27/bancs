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

class ReliabilityBenchmark:
    """
    TODO: docstring
    """
    def __init__(self, problems, methods=["BANCS", "SS", "NAIS"], sizes=[int(1e4)]):
        self.sizes = sizes
        self.methods = methods
        self.problems = problems
        problem_names = [problem.getName() for problem in problems]

        multi_indexes = pd.MultiIndex.from_product([problem_names, methods, sizes], names=["problem", "method", "size"])
        self.df_results = pd.DataFrame([], columns=["pf_mean", "pf_ref", "pf_std", "pf_med", "pf_ic_low", "pf_ic_up", "pfs"], index=multi_indexes)
        self.df_results['pfs'].astype('object')

        self.p0 = 0.1

    def run(self, reps=10, m="beta", save_file=None):
        for problem, method, size in product(self.problems, self.methods, self.sizes):
            if m == "beta":
                m = int(size * self.p0)
            elif m == "amise":
                dim = problem.getEvent().getAntecedent().getDimension()
                m = int(1 + np.floor(size ** (2 / (4 + dim))))

            pf_ref = problem.getProbability()
            pfs = np.array([])
            for rep in range(reps):
                #pf = run BANCS
                if method=="BANCS":
                    bacs = BANCS(problem.getEvent(), N=size, M=m, p0=self.p0)
                    _ = bacs.run()
                    pf = bacs.compute_pf()

                elif method=="SS":
                    ss = ot.SubsetSampling(problem.getEvent())
                    ss.setMaximumOuterSampling(size)
                    ss.setMaximumCoefficientOfVariation(-1.0)
                    ss.setConditionalProbability(self.p0)
                    ss.setBlockSize(10)
                    timer = ot.TimerCallback(120)
                    ss.setStopCallback(timer)
                    ss.run()
                    pf = ss.getResult().getProbabilityEstimate()

                elif method=="NAIS":
                    nais = ot.NAIS(problem.getEvent(), self.p0)
                    nais.run()
                    nais.setMaximumOuterSampling(size)
                    nais.setMaximumCoefficientOfVariation(-1.0)
                    timer = ot.TimerCallback(10)
                    nais.setStopCallback(timer)
                    pf = nais.getResult().getProbabilityEstimate()

                pfs = np.append(pfs, [pf])
            pf_ic = bootstrap((pfs,), np.mean, confidence_level=0.95)
            self.df_results.loc[(problem.getName(), method, size)] = [np.mean(pfs), 
                                                                        pf_ref,
                                                                        np.std(pfs),
                                                                        np.median(pfs),
                                                                        pf_ic.confidence_interval.low, 
                                                                        pf_ic.confidence_interval.high, 
                                                                        pfs.tolist()
                                                                        ]
            print(f"DONE: {problem.getName()}, {method}, {size:.0E}")
            
            if save_file is not None:
                self.df_results.to_csv(save_file, index=True)