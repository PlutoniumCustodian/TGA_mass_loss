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
import matplotlib.pyplot as plt
from statistics import mean

# get varialble from pickle jar
Meta_df = pd.read_pickle("pickle_jar/Meta_df.pkl")

lst_TGA_data= pd.read_pickle("pickle_jar/TGA_ex_data.pkl") #creates list of
#data frames each data frame has one TGA scan

# Check that you have the same number of data files and meta files
if len(lst_TGA_data) != len(Meta_df):
    print('Error meta and expermental data do not match')
#%% Find values for the weight at T of interest

Num_of_records = len(lst_TGA_data)
Mass = [0]*Num_of_records #make list of "0"s with same length as data
Tmin = 30
Tmax =1000
TRange = range(Tmin, Tmax+1) #make a list with one interger for 30-1000C
IntigerT = list(TRange)
IntigerM = [0]*len(IntigerT)
df_int_mass = pd.DataFrame({'Temperature':IntigerT})
#Get TGA data out of the list containing dataframes
for x in range(Num_of_records):

df = lst_TGA_data[x] #gets one TGA rocord
TGA_T = np.array(df.loc[:,'Temperature'])
TGA_M = np.array(df.iloc[:, 2]) # call by index because the are 2 "weight" column

for y in TRange:
# find index of first time T (y) reaches T-0.5
    inx1 = next(n for n, val in enumerate(TGA_T)if val > IntigerT[y]-0.5 )
    inx2 = next(n for n, val in enumerate(TGA_T)if val > IntigerT[y]+0.5 )-1
    IntigerM[(y-Tmin)] = mean(TGA_M[inx1:inx2])
df_int_mass.insert(x,'mass', IntigerM)


# # intMass = np.array(Meta_df.loc[:,'Initial_mass_TGA_mg'])
# Mass_df = pd.DataFrame({'Intial_mass_external': np.array(Meta_df.loc[:,'Intial_mass_external_mg']),
#                        'Initial_mass_TGA': np.array(Meta_df.loc[:,'Initial_mass_TGA_mg'])})
# Mass_df['Initial_mass_TGA'] = Mass_df['Initial_mass_TGA'].astype(float, errors = 'raise')
# Mass_df['30C'] = M30
# Mass_df['40C'] = M40
# Mass_df['105C'] = M105
# Mass_df['180C'] = M180
# Mass_df['550C'] = M550
# Mass_df['1,000C'] =M1000
#%% Export data frame
#Mass_df has the mass at select T for all of the "good" data that was
#imported in the TGA_Mass_loss_data_import
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

#%% get chem info based on "PC name"

#Get chemistry of names cement precoursors atomic % of metal x makes of total
# metalic content  
pc_chem = pd.read_csv('input_data/Atomic_fraction_of_metals_in_PC.csv')

# get labes of above data and make empty df with lenght that matches data
chem_lable = list(pc_chem)
chem_lable = chem_lable[1:]
dat_length = range(len(Meta_df))
chem_length =range(len(chem_lable)-1) # range of lenth = number of data labes in pc_chem

chem_df = pd.DataFrame(columns=[chem_lable], index=dat_length)
chem_df.columns = chem_lable
r_name = range(len(pc_chem))

# using the 'PC name" asign vlaues from pc_chem to chem_df
for x in dat_length:
    for z in r_name:
        if Meta_df.loc[x,'PC_Name'] == pc_chem.loc[z,'Name']:
            for y in [0,1,2,3]:
                chem_df.iloc[x,y] = pc_chem.iloc[z,(y+1)]
    
#%% get mineral info (Metalic content as represented as oxide used in cement 
# shorthand) based on "PC name"
pc_mineral = pd.read_csv('input_data/Weight_percent_oxides_in_PC.csv')

# get labes of above data and make empty df with lenght that matches data
min_lable = list(pc_mineral)
min_lable = min_lable[1:] # removeds first row form list
dat_length = range(len(Meta_df))
mineral_length =range(len(min_lable)-1) # range of lenth = number of data labes in pc_chem

mineral_df = pd.DataFrame(columns=[min_lable], index=dat_length)
mineral_df.columns = min_lable
r_name = range(len(pc_mineral))

# using the 'PC name" asign vlaues from pc_mineral to chem_df
for x in dat_length:
    for z in r_name:
        if Meta_df.loc[x,'PC_Name'] == pc_mineral.loc[z,'Name']:
            for y in [0,1,2,3]:
                mineral_df.iloc[x,y] = pc_mineral.iloc[z,(y+1)]






#%% Export more data frame
# # fractional weight loss for teperature ranges of interest
# fl_df.to_csv("output_data/TGA_fractional_weight_loss.csv")
# pd.to_pickle(fl_df,"pickle_jar/TGA_fractional_weight_loss.pkl" )
# # has the target mole percent of Mg, Al, Si, & P for each sample
# pd.to_pickle(chem_df, "pickle_jar/TGA_target_chem_comp.pkl")
# pd.to_pickle(mineral_df, "pickle_jar/ICP_value_for_TGA_dat.pkl")

#%% Split out data sets


# get list of file numbers for the data set that of samples activeate with
# 1ml 10M NaOH per gram of cement precourser  aged for 24hrs at 35C before drying
Index_set_1ml_1g = pd.read_csv('input_data/Sample_set_1ml_per_1gram.csv')

# creates df with the rows matching the index (file number) improrted in the above lines
Data_set_1ml_1g = fl_df.loc[Index_set_1ml_1g['File number']]
Meta_set_1ml_1g = Meta_df.loc[Index_set_1ml_1g['File number']]
Chem_set_1ml_1g = chem_df.loc[Index_set_1ml_1g['File number']]

# get list of file numbers for the data set that of samples activeate with
# 1ml and 0.5ml 10M NaOH per gram of cement precourser  aged for 24hrs at 35C before drying
# at time of progam is all of the good TGA data we have
Index_set_good = pd.read_csv('input_data/Sample_set_1ml_and_halfml_per_1gramPC.csv')

# creates df with the rows matching the index (file number) improrted in the above lines
Data_set_good = fl_df.loc[Index_set_good['File number']]
Meta_set_good = Meta_df.loc[Index_set_good['File number']]
Mineral_set_good = mineral_df.loc[Index_set_good['File number']]

#%% Export even more data frames

# #Data set with all the 1mL o.5ml and Centroid v. NaOH concentration data
# #For samples activated with NaOH
# pd.to_pickle(Data_set_good, "pickle_jar/TGA_data_good.pkl") # wieght loss data
# pd.to_pickle(Meta_set_good, "pickle_jar/Meta_set_good.pkl")
# pd.to_pickle(Mineral_set_good, "pickle_jar/Mineral_set_good.pkl")            

#%% Make some plots

legspot = 'upper center' # Determines where legend is placed

# font = FontProperties()
# font.set_family('sans-serf')
# font.set_name('Arial')
# font.set_size(9)
n=0

mktype = ["o","v","s","X"]
fl_lables = list(Data_set_1ml_1g)
for n in [0,1,2,3]:
    fig, ax = plt.subplots(figsize=(7.08,5)) #size is in inches
    for m in [0,1,2,3]:
        ax.plot(Chem_set_1ml_1g.iloc[:,n] * 100, Data_set_1ml_1g.loc[:,fl_lables[m+4]]*100,\
                marker=mktype[m],fillstyle='none', ms=5,ls='', label=fl_lables[m+4])
    ax.set_xlabel(chem_lable[n] + " (atm. % of total metal)", fontsize=9)
    ax.set_ylabel("Weight loss (%)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc=legspot)
    ax.set_xlim([5,65])
    ax.set_ylim([0,50])
    svg_name_path = 'output_data/figures/' + chem_lable[n] + '_wt_loss.svg'
    # fig.savefig(svg_name_path, transparent=False, bbox_inches="tight")

#%% Do some stats

# Make df with loose bound and unbound water 
# then calculate Pearson Correlation Coeffiecient
df_luwat = fl_df.loc[:,['Unbound_water','loose_bound_water']] #, ]
print(df_luwat.corr())

#%% Figure some stuff out

fl_df.hist()

