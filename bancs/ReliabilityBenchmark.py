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
from tqdm.contrib.itertools import product
from scipy.stats import bootstrap
from multiprocessing import Pool

class ReliabilityBenchmark:
    """
    TODO: docstring
    """
    def __init__(self, problems, methods=["BANCS", "SS", "NAIS"], sizes=[int(1e4)], ebc_tuning="amise", reps=10):
        self.sizes = sizes
        self.methods = methods
        self.problems = problems
        problem_names = [problem.getName() for problem in problems]

        multi_indexes = pd.MultiIndex.from_product([problem_names, methods, sizes], names=["problem", "method", "size"])
        self.df_results = pd.DataFrame([], columns=["nb_samples", "pf_mean", "pf_ref", "pf_std", "pf_ic_low", "pf_ic_up", "pfs"], index=multi_indexes)
        self.df_results['pfs'].astype('object')
        self.p0 = 0.1
        self.ebc_tuning = ebc_tuning
        self.reps = reps

    def _run_one_rep(self, problem, method, size, m, seed):
        ot.RandomGenerator.SetSeed(seed)
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
            ss.setMaximumTimeDuration(120)
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
                nais.setMaximumTimeDuration(120)
                nais.run()
                res = nais.getResult()
                nb_samples = res.getOuterSampling()
                pf = res.getProbabilityEstimate()
                pf_std = res.getStandardDeviation()
        return pf, nb_samples

    def run(self, save_file=None):
        for problem, method, size in product(self.problems, self.methods, self.sizes, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            pf_ref = problem.getProbability()
            pfs = np.array([])
            nb_samples_list = np.array([])

            repeated_setup = np.array([[problem, method, size, self.ebc_tuning]] * self.reps)
            repeated_setup = np.concatenate((repeated_setup, np.arange(self.reps).reshape(-1, 1)), axis=-1)
            if method == "BANCS":
                with Pool(10) as p:
                    results = p.starmap(self._run_one_rep, repeated_setup)
            else : 
                results = []
                for rep in repeated_setup: 
                    results.append(self._run_one_rep(*rep))
            results = np.array(results)
            # Results
            pfs = results[:, 0]
            nb_samples_list = results[:, 1]
            pf_ic = bootstrap((pfs,), np.mean, confidence_level=0.95)
            pf_low = pf_ic.confidence_interval.low
            pf_up = pf_ic.confidence_interval.high
            pf_std = np.std(pfs)

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