# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:18:39 2022

@author: Titus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import time

#%% Import dat from pickle jar

TGA_data = pd.read_pickle("pickle_jar/TGA_data_good.pkl") # weight loss data
Meta_data = pd.read_pickle("pickle_jar/Meta_set_good.pkl")
Mineral_data = pd.read_pickle("pickle_jar/Mineral_set_good.pkl") #ICP results

#%% Make df with the data needed for subset selection
dat = pd.concat([Mineral_data, TGA_data], axis=1)
dat['Concentration(mM)'] = Meta_data.loc[:,'Concentration(mM)']
dat['L/S'] = Meta_data.loc[:,'L/S']
# dat = dat.apply(pd.to_numeric) #converts all values to numbers

#%% pick x and y data

X= dat[[('MgO',), ('Al2O3',), ('SiO2',), ('P2O5',), 'Concentration(mM)', 'L/S']]
y = dat[['Bound_water']]

#%% Try models with 1 predictor all the way to p predictors.
ct = 0
best_models = []

tic = time.time()

for p in range(1, len(X.columns)+1):

    result_mod_p = []
    for comb in itertools.combinations(X.columns, p):

        # Fit the model with 'comb' predictors and store the result
        mod = sm.OLS(y, X[list(comb)]).fit() # Note no intercept included by OLS since we are not using formula.api 
        result_mod_p.append({'model': mod, 'ssr': mod.ssr}) # Note that mod.ssr is equivalent to ((mod.predict( X[list(comb)] ) - y['CoolElec']) ** 2).sum()
        ct += 1

    # Select and store the best model with p predictors
    df_p = pd.DataFrame(result_mod_p)
    best_p_model = df_p.loc[ df_p['ssr'].argmin() ]
    best_models.append( best_p_model )
    
    # Print the variable names for the best p-parameter model
    x_list = ', '.join( list(best_p_model['model'].params.index) )
    print('Best model with', str(p), 'predictor(s) includes:', x_list+'.')
    
toc = time.time()
print(f'Considered {str(ct)} models in {(toc-tic):1.4f} seconds.')

# Store some of the stats so we can compare to other methods
method_compare = [pd.DataFrame([[ct,f'{(toc-tic):1.4f}']], columns=['# Models','Time(s)'], index=pd.Index(['Exhaustive']))]