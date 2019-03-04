# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:58:24 2019

@author: vlade

This script will allow the user to build a fitted yield curve according to 
a number of methods. Currently this implementation only allows for one method:
the Nelson Siegal method as in the Diebold Li (2006) paper.
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin
from scipy.optimize import least_squares


class NelsonSiegal:
    
    def __init__(self, est_l = True):
        self.factors = None
        self.est_l = est_l
        pass
    
    def fit(self, data, l, params=[1, 1, 1, 1]):
        """ Here we begin by defining the function we want to minimise (fp),
        then define the cost function (e) and finally create an array with 
        the maturities (x).
        This function also starts with the option to estiamte lambda param
        or supply own.
        If you choose to estimate lambda the optimisation algorithm will run
        a non-linear least squares. If you choose to supply a lambda estimate,
        the program will run a standard least squares minimisation algo.
        It seems that the nonlinear optimisation is highly dependent on
        the starting parameters provided, use this with care."""
        
        if self.est_l:
            fp = lambda c, x: c[0] + (c[1]*((1-np.exp(-c[3]*x))/(c[3]*x))) + (c[2]*(((1-np.exp(-c[3]*x))/(c[3]*x))-np.exp(-c[3]*x)))
            e = lambda p, x, y: ((fp(p,x)-y)**2).sum()
            x = np.array([float(i) for i in data.columns])
            
            factors = pd.DataFrame(columns = ["Factor 1",
                                              "Factor 2",
                                              "Factor 3",
                                              "Lambda"])
            
            for index, row in data.iterrows():
                p0 = np.array(params)  # initial parameter values
                p = least_squares(e, p0, args=(x,row),
                             method="trf", jac="2-point") # fitting the data with least_squares
                p = p.x.reshape(1,4) 
                temp = pd.DataFrame(p, columns = ["Factor 1",
                                                  "Factor 2",
                                                  "Factor 3",
                                                  "Lambda"])
                factors = factors.append(temp)
    
            factors.index = data.index
            self.factors = factors
        
        else:
            fp = lambda c, x: c[0] + (c[1]*((1-np.exp(-l*x))/(l*x))) + (c[2]*(((1-np.exp(-l*x))/(l*x))-np.exp(-l*x)))
            e = lambda p, x, y: ((fp(p,x)-y)**2).sum()
            x = np.array([float(i) for i in data.columns])
            
            factors = pd.DataFrame(columns = ["Factor 1",
                                              "Factor 2",
                                              "Factor 3",])
            
            for index, row in data.iterrows():
                p0 = np.array(params)  # initial parameter values
                p = fmin(e, p0, args=(x,row)).reshape(1,3) # fitting the data with fmin
                temp = pd.DataFrame(p, columns = ["Factor 1",
                                                  "Factor 2",
                                                  "Factor 3"])
                factors = factors.append(temp)
    
            factors.index = data.index
            self.factors = factors            
    
    def build_curve(self, x):
        yldDf = pd.DataFrame(columns = [float(i) for i in x])
       
        for i in x: 
            yldList = []        
            for index, row in self.factors.iterrows(): 
                yldList.append(row[0] + (row[1]*((1-np.exp(-row[3]*i))/(row[3]*i))) + (row[2]*(((1-np.exp(-row[3]*i))/(row[3]*i))-np.exp(-row[3]*i))))                   
            yldDf[i] = pd.Series(yldList)
            
        return yldDf