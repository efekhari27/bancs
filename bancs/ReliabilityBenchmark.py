#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot
import bancs as bc
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
        self.df_results = pd.DataFrame([], columns=["nb_samples", "pf_mean", "pf_ref", "pf_std", "pf_ic_low", "pf_ic_up", "pfs"], index=multi_indexes)
        self.df_results['pfs'].astype('object')

        self.p0 = 0.1

    def run(self, reps=10, m="amise", save_file=None):
        for problem, method, size in product(self.problems, self.methods, self.sizes):
            dim = problem.getEvent().getAntecedent().getDimension()
            if m == "amise":
                m = int(1 + np.floor(size ** (2 / (4 + dim))))

            pf_ref = problem.getProbability()
            pfs = np.array([])
            nb_samples_list = np.array([])
            
            for rep in range(reps):
                #pf = run BANCS
                if method=="BANCS":
                    if problem.name=="Oscillator":
                        
                        bancs = bc.BANCS(problem.getEvent(), N=size, M=m, p0=self.p0, lower_truncatures=[0.] * 8)
                    else:
                        bancs = bc.BANCS(problem.getEvent(), N=size, M=m, p0=self.p0)
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
                    if problem.name=="Oscillator":
                        nais = bc.NAISAlgorithm(event=problem.getEvent(), n_IS=size, rho_quantile=self.p0, lower_truncatures=[0.] * 8)
                        nais.run()
                        NAIS_result = nais.getResult()
                        pf = NAIS_result.getProbabilityEstimate()
                        pf_std = 0.01
                        nb_samples = nais.nb_subsets * size
                    else:
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
                pfs = np.append(pfs, [pf])
                nb_samples_list = np.append(nb_samples_list, [nb_samples])
            
            if reps > 1: 
                pf_ic = bootstrap((pfs,), np.mean, confidence_level=0.95)
                pf_low = pf_ic.confidence_interval.low
                pf_up = pf_ic.confidence_interval.high
                pf_std = np.std(pfs)
            else : 
                pf_low = pf - 2 * pf_std
                pf_up = pf + 2 * pf_std
            self.df_results.loc[(problem.getName(), method, size)] = [np.median(nb_samples_list), 
                                                                        np.mean(pfs), 
                                                                        pf_ref,
                                                                        pf_std,
                                                                        pf_low, 
                                                                        pf_up, 
                                                                        pfs.tolist()
                                                                        ]
            print(f"DONE: {problem.getName()}, {method}, {size:.0E}")
            
        if save_file is not None:
            self.df_results.to_csv(save_file, index=True)