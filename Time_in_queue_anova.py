# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:41:16 2022

@author: Titus
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import bartlett
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
#%% Import dat from pickle jar

TGA_data = pd.read_pickle("pickle_jar/TGA_data_good.pkl") # weight loss data
Meta_data = pd.read_pickle("pickle_jar/Meta_set_good.pkl")
Mineral_data = pd.read_pickle("pickle_jar/Mineral_set_good.pkl") #ICP results
run_order = pd.read_pickle("pickle_jar/Run_order.pkl")

#%% Make df with all the data
dat = pd.concat([Mineral_data, TGA_data], axis=1)
dat['Concentration(mM)'] = Meta_data.loc[:,'Concentration(mM)']
dat['L/S'] = Meta_data.loc[:,'L/S']
dat['PC_name'] = Meta_data.loc[:,'PC_Name']
dat['run_number'] = run_order.loc[:,'run_number']

#%% Split out the data by run number

run_order = pd.DataFrame()
run_order['Pan_#'] = Meta_data.loc[:,'Pan_#']
run_order['run_number'] = 0

#List for eahc group of Pan_#
A=[]
B=[]
C=[]
Yname = 'Tight_bound_water'
for i in list(dat.index.values) :
    if dat.loc[i,'run_number'] == 1:
        A.append(dat.loc[i,Yname])
    if dat.loc[i,'run_number'] == 2:
        B.append(dat.loc[i,Yname])
    if dat.loc[i,'run_number'] == 3:
        C.append(dat.loc[i,Yname])

#%% generate Q-Q plot of data vs normaly distributed data



fig = sm.qqplot(np.array(A), line='45', fit=True)
plt.title('A')
plt.show()

fig = sm.qqplot(np.array(B), line='45', fit=True)
plt.title('B')
plt.show()

fig = sm.qqplot(np.array(C), line='45', fit=True)
plt.title('C')
plt.show()


#%% Conduct the one-way ANOVA

# Bartlett’s Test is a statistical test that is used to determine whether
# or not the variances between several groups are equal.
# If P-value is larger than your alpha value,there not enough evidance
# to say variance is diferent, so you can use one-way ANOVA
stat, p = bartlett(A, B, C)


print("bartlett test P-value",p)

alpha = 0.05
if p > alpha:
    print("Variance is the same of normal,  one-way ANOVA is valid")
else:
    print("Variance is NOT the same one-way ANOVA not valid")
    
print(f_oneway(A, B, C))
print("When P is less than alpha, at lest one poulation mean is different")

# Kruskal-Wallis test can be used whe normality assumption is violated
print(stats.kruskal(A, B, C))
