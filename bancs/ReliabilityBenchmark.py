#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot

class ReliabilityBenchmark:
    """
    TODO: docstring
    """
    def __init__(self, methods, problems, sizes=[int(1e4)]):
        self.methods = methods
        self.problems = problems
        self.sizes = sizes