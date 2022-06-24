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
from matplotlib import gridspec

#%% Import dat from pickle jar

TGA_data = pd.read_pickle("pickle_jar/TGA_data_good.pkl") # weight loss data
Meta_data = pd.read_pickle("pickle_jar/Meta_set_good.pkl")
Mineral_data = pd.read_pickle("pickle_jar/Mineral_set_good.pkl") #ICP results

#%% Make df with the data needed for subset selection
dat = pd.concat([Mineral_data, TGA_data], axis=1)
dat['Concentration(mM)'] = Meta_data.loc[:,'Concentration(mM)']
dat['L/S'] = Meta_data.loc[:,'L/S']
dat = dat.apply(pd.to_numeric) #converts all values to numbers

#%% pick x and y data

X= dat[['MgO', 'Al2O3', 'SiO2', 'P2O5', 'Concentration(mM)', 'L/S']]
y = dat[['Tight_bound_water']] #change this to change your responce variable

#%% Try models with 1 predictor all the way to p predictors.
ct = 0
best_models = []
best_x_list = []
all_models = []
mod_name = []
var_ct = [] #Used to count index the number of variables in a model
tic = time.time()
X = sm.add_constant(X)
for p in range(1, (len(X.columns)+1)):

    result_mod_p = []
    for comb in itertools.combinations(X.columns, p):

        # Fit the model with 'comb' predictors and store the result
        mod = sm.OLS(y, X[list(comb)]).fit() # Note no intercept included by OLS since we are not using formula.api 
        result_mod_p.append({'model': mod, 'ssr': mod.ssr}) # Note that mod.ssr is equivalent to ((mod.predict( X[list(comb)] ) - y['CoolElec']) ** 2).sum()
        all_models.append(mod)
        var_ct.append(p)
        mod_name.append(mod.params.index)
        ct += 1

    # Select and store the best model with p predictors
    df_p = pd.DataFrame(result_mod_p)
    best_p_model = df_p.loc[ df_p['ssr'].argmin() ]
    best_models.append( best_p_model )
    
    #store all the model info for later
   
    
    # Print the variable names for the best p-parameter model
    x_list = ', '.join( list(best_p_model['model'].params.index) )
    print('Best model with', str(p), 'predictor(s) includes:', x_list+'.')
    best_x_list.append(x_list) #store the printed info to a list
toc = time.time()
print(f'Considered {str(ct)} models in {(toc-tic):1.4f} seconds.')

# Store some of the stats so we can compare to other methods
method_compare = [pd.DataFrame([[ct,f'{(toc-tic):1.4f}']],\
                               columns=['# Models','Time(s)'], index=pd.Index(['Exhaustive']))]
    
#%% Extract some model stats so we can compare.
num_p_B = range(1, len(best_models)+1)
ssr_B = [ best_models[i]['model'].ssr for i in range(len(best_models)) ]
r2_B = [ best_models[i]['model'].rsquared for i in range(len(best_models)) ]
r2adj_B = [ best_models[i]['model'].rsquared_adj for i in range(len(best_models)) ]
aic_B = [ best_models[i]['model'].aic for i in range(len(best_models)) ]
bic_B = [ best_models[i]['model'].bic for i in range(len(best_models)) ]


#%% Extract some model stats for all models so we can compare.
ct2 = range(len(all_models))
num_p = range(1, len(all_models)+1)
ssr = [ all_models[i].ssr for i in ct2]
r2 = [ all_models[i].rsquared for i in ct2]
r2adj = [ all_models[i].rsquared_adj for i in ct2]
aic = [ all_models[i].aic for i in ct2]
bic = [ all_models[i].bic for i in ct2]

mod_stat_df = pd.DataFrame({'x_values':mod_name})
mod_stat_df['# of Var'] = var_ct
mod_stat_df['ssr'] = ssr
mod_stat_df['r^2'] = r2
mod_stat_df['r^2adj'] = r2adj
mod_stat_df['AIC'] = aic
mod_stat_df['BIC'] = bic

#Warning writes file
# mod_stat_df.to_csv("output_data/xxx_best_subset_linear_regression_and_C.csv")

#%% Create a plotting function to plot the stats
def plot_stat(x, y, x_b, y_b, name, ax):
    # x and y are the best fit for each # of prdictors 
    #data and x_b and y_b are the other models
    ax.grid(True)
    ax.plot(x, y, color = 'b', marker = 'o', label=name,)
    ax.plot(x_b, y_b, color = 'k', marker = 'o', label=name, ls='')
    ax.plot(np.argmax(y)+1, max(y), color = 'm', marker = 'D', markersize = '13' ) # label max point
    ax.plot(np.argmin(y)+1, min(y), color = 'c', marker = 'D', markersize = '13') # label min point
    ax.axhline(y=min(y), color='c', linestyle='-.') #min line for ref
    ax.axhline(y=max(y), color='m', linestyle='-.') #max line for ref
    ax.set_xlabel('# of Predictors')
    ax.set_ylabel(name)
    ax.set_xlim([0,p])
    
# %%Plot the sum of squared residuals and Rsquared
# not currently working
fig = plt.figure(figsize=(20,30))
plt.rcParams.update({'font.size': 16})
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

plot_stat(num_p_B, r2_B, var_ct, r2, 'R$^2$', plt.subplot(gs[0]))
plot_stat(num_p_B, r2adj_B, var_ct, r2adj, 'R$^2$ adjusted', plt.subplot(gs[1]))
plot_stat(num_p_B, bic_B, var_ct, bic, 'BIC', plt.subplot(gs[2]))
plot_stat(num_p_B, aic_B, var_ct, aic, 'AIC', plt.subplot(gs[3]))
plot_stat(num_p_B, ssr_B, var_ct, ssr, 'SSR', plt.subplot(gs[4]))

#Warning writes file
# fig.savefig('output_data/xxx_best_subset_linear_regression.svg', transparent=False, bbox_inches="tight")

#%% Gett details of best model

# reportFile = open('output_data/xxx_Best_OLR_.txt', 'w')
# print(all_models[86].summary(), file=reportFile)
# reportFile.close()

# #Use df to store info from summary
# df1 = pd.read_html(all_models[44].summary().tables[0].as_html(),header=0,index_col=0)[0]
# df2 = pd.read_html(all_models[44].summary().tables[1].as_html(),header=0,index_col=0)[0]
# df3 = pd.read_html(all_models[44].summary().tables[2].as_html(),header=0,index_col=0)[0]
# print(df1)
# print(df2)
# print(df3)