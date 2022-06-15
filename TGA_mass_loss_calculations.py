# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:39:16 2022

@author: Titus
"""

#%% Intial setup
# import os
# import re
import pandas as pd
import numpy as np

# get varialble from pickle jar
Meta_df = pd.read_pickle("pickle_jar/Meta_df.pkl")

lst_TGA_data= pd.read_pickle("pickle_jar/TGA_ex_data.pkl")

# Check that you have the same number of data files and meta files
if len(lst_TGA_data) != len(Meta_df):
    print('Error meta and expermental data do not match')
#%% Calculate all the differnt weight loss

Num_of_records = len(lst_TGA_data)
M30 = [0]*Num_of_records #make list of "0"s with same length as data
M40 = [0]*Num_of_records
M105 = [0]*Num_of_records
M180 = [0]*Num_of_records
M550 = [0]*Num_of_records
M1000 = [0]*Num_of_records

#Get TGA data out of the list containing dataframes
for x in range(Num_of_records):
    df = lst_TGA_data[x]
    TGA_T = np.array(df.loc[:,'Temperature'])
    TGA_M = np.array(df.iloc[:, 2]) # call by index because the are 2 "weight" column

    # find index of first time T reaches 30C
    inx30 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 30 )
    M40[x] = TGA_M[inx30]
    inx40 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 40 )
    M30[x] = TGA_M[inx30]
    # find index of first time T reaches 105C
    inx105 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 105 )
    M105[x] = TGA_M[inx105]
    # find index of first time T reaches 180C
    inx180 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 180 )
    M180[x] = TGA_M[inx180]
    # find index of first time T reaches 550C
    inx550 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 550 )
    M550[x] = TGA_M[inx550]
    M1000[x] = TGA_M[-1] 

# intMass = np.array(Meta_df.loc[:,'Initial_mass_TGA_mg'])
Mass_df = pd.DataFrame({'Intial_mass_external': np.array(Meta_df.loc[:,'Intial_mass_external_mg']),
                       'Initial_mass_TGA': np.array(Meta_df.loc[:,'Initial_mass_TGA_mg'])})
Mass_df['Initial_mass_TGA'] = Mass_df['Initial_mass_TGA'].astype(float, errors = 'raise')
Mass_df['30C'] = M30
Mass_df['40C'] = M40
Mass_df['105C'] = M105
Mass_df['180C'] = M180
Mass_df['550C'] = M550
Mass_df['1,000C'] =M1000
#%% Export data frame
Mass_df.to_csv("output_data/TGA_Mass_at_selct_T.csv")
pd.to_pickle(Mass_df,"pickle_jar/Mass_df.pkl" )

#%% calculate weight loss

# wieght lost siting in air waiting to be loaded into TGA
# (this also includes diferance in persion and acuracy of scales)
Mint = np.array(Meta_df.loc[:,'Intial_mass_external_mg'])
Mtga = np.array(Meta_df.loc[:,'Initial_mass_TGA_mg'])
# Mtga = Mtga.astype(float)

# mass loss in mg
ml_air = Mass_df['Intial_mass_external'] - Mass_df['Initial_mass_TGA']
# fractions mass losss
fl_air = ml_air / Mass_df['Intial_mass_external']

#drying of sample during TGA furnace purge
fl_purge = (Mass_df['Initial_mass_TGA'] - Mass_df['40C']) / Mass_df['Initial_mass_TGA']

fl_dry = (Mass_df['Intial_mass_external'] - Mass_df['40C']) /  Mass_df['Intial_mass_external']

# Unbound water
fl_pore = (Mass_df['40C'] - Mass_df['105C']) / Mass_df['40C']

# Bound water
fl_bound = (Mass_df['105C'] - Mass_df['1,000C']) / Mass_df['40C']

# Loose bound water
fl_loose = (Mass_df['105C'] - Mass_df['180C']) / Mass_df['40C']
# Tightly bound water
fl_tight = (Mass_df['180C'] - Mass_df['550C']) / Mass_df['40C']
# total post drying weight loss

fl_df = pd.DataFrame({'NEUP_Sample_#': Meta_df.loc[:,'NEUP_Sample_#']})
fl_df['drying_in_air'] = fl_air
fl_df['Ar_pruge'] = fl_purge
fl_df['All_Pre-scan_drying'] = fl_dry 
fl_df['Unbound_water'] = fl_pore
fl_df['Bound_water'] = fl_bound 
fl_df['loose_bound_water'] = fl_loose 
fl_df['Tight_bound_water'] = fl_tight

#%% Export more data frame
fl_df.to_csv("output_data/TGA_fractional_weight_loss.csv")
pd.to_pickle(fl_df,"pickle_jar/TGA_fractional_weight_loss.pkl" )