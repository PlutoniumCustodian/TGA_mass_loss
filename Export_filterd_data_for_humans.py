# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:30:09 2022
This script takes the TGA data that was manualy and saved as pickles adn saves
to csv, this data includes all (as of 9/2022) good data for samples aged 24hr
with 1g PC/ 1ml 10M NaOH, & 0.5g PC/ 1ml 10M NaOH

@author: Titus
"""

import pandas as pd


#%% Import dat from pickle jar

TGA_data = pd.read_pickle("pickle_jar/TGA_data_good.pkl") # weight loss data
Meta_data = pd.read_pickle("pickle_jar/Meta_set_good.pkl")
Mineral_data = pd.read_pickle("pickle_jar/Mineral_set_good.pkl") #ICP results

#%% Make df with the data needed for subset selection
dat = pd.concat([Mineral_data, TGA_data], axis=1)
dat['Concentration(mM)'] = Meta_data.loc[:,'Concentration(mM)']
dat['L/S'] = Meta_data.loc[:,'L/S']
dat['PC_name'] = Meta_data.loc[:,'PC_Name']
#dat = dat.apply(pd.to_numeric) #converts all values to numbers
dat.to_csv("output_data/22_09_01data.csv")
