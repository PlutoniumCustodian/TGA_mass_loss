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
M30 = [0]*Num_of_records
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
Mass_df['30C'] = M30
Mass_df['105C'] = M105
Mass_df['180C'] = M180
Mass_df['550C'] = M550
Mass_df['1,000C'] =M1000
