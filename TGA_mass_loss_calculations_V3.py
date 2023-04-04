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
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
import plotly.express as px
# from statsmodels.regression.rolling import RollingWLS

# get varialble from pickle jar
Meta_df = pd.read_pickle("pickle_jar/Meta_df.pkl")

lst_TGA_data= pd.read_pickle("pickle_jar/TGA_ex_data.pkl") #creates list of
#data frames each data frame has one TGA scan

# Check that you have the same number of data files and meta files
if len(lst_TGA_data) != len(Meta_df):
    print('Error meta and expermental data do not match')
#%% Bin mass of sample vs T into 1C wide bins

Num_of_records = len(lst_TGA_data)
Mass = [0]*Num_of_records #make list of "0"s with same length as data
Tmin = 30
Tmax =1000
TRange = range(Tmin, Tmax+1) #make a list with one interger for 30-1000C
IntigerT = list(TRange)
lst_edges = np.arange(Tmin-0.5 , Tmax +1.5 , 1)
# lst_edges = lst_edges + 0.5 #make edges so bin centered intiger
IntigerM = [0]*len(IntigerT)
df_bin_mass = pd.DataFrame(index=IntigerT) #index is now degrees C
#Get TGA data out of the list containing dataframes
for x in range(Num_of_records):
    df = lst_TGA_data[x] #gets one TGA rocord
    TGA_T = np.array(df.loc[:,'Temperature'])
    TGA_T = np.ravel(TGA_T[:,0])
    TGA_M = np.array(df.iloc[:, 2]) # call by index because the are 2 "weight" column
    
    bin_means, bin_edges, binnumber=stats.binned_statistic(TGA_T, TGA_M, bins=lst_edges)
    
    df_bin_mass[x] = bin_means
    
    # for y in TRange:
    # # find index of first time T (y) reaches T-0.5
    #     inx1 = next(n for n, val in enumerate(TGA_T)if val > IntigerT[y]-0.5 )
    #     inx2 = next(n for n, val in enumerate(TGA_T)if val > IntigerT[y]+0.5 )-1
    #     IntigerM[(y-Tmin)] = mean(TGA_M[inx1:inx2])
    # df_int_mass.insert(x,'mass', IntigerM)


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

#%% Normalize weight loss to percent of dry weight (dry is 105C)
df_percent_dry_weight = pd.DataFrame(index=IntigerT) #index is now degrees C
M105_a = df_bin_mass.loc[105]
# M105_a.reset_index(drop=True,inplace=True)
# M105_b = M105_a.transpose()
# M105_df = pd.Series(M105_series, index=df_bin_mass.columns)
# M105_arry = M105_df.to_numpy()
df_percent_dry_weight = df_bin_mass.div(M105_a, axis='columns')

#%% calculate derivitave weight loss via Rolling window OLS (mg/degree)
# This one is not normalized do not use it
# window = 7
# df_dM_dT = pd.DataFrame({'Temperature':IntigerT})
# dT_min = int((window/2)-0.5) #calculates the number of indexies to shift data
# # so that the value return is for the midle of the window
# dT_max = Tmax-dT_min
# dat_shift= list(range(0,dT_min)) # makes list of rows to delet later
# # df_dM_dT = pd.DataFrame({'Temperature' : TRange[dT_min:dT_max]})
# for x in range(Num_of_records):
#     endog = df_bin_mass[x] # this is the y values for  
#     exog = sm.add_constant(IntigerT)
#     # exog = sm.add_constant(IntigerT, prepend=False)
#     mod = RollingOLS(endog, exog, window=window)
#     fitted = mod.fit()
#     temp = fitted.params
#     temp=temp.drop(dat_shift) # drops first few values so that the window is for the center of the window instead of
#     # an edge
#     temp=temp.reset_index(drop=True) # resets index after deleting rows
#     df_dM_dT[x] = temp['x1']
    
# #code to drop first 3 and last 3 rows
# last=df_dM_dT.last_valid_index()
# allmost_last = last - dT_min
# end_drop = list(range(allmost_last , last))
# dat_drop = dat_shift + end_drop
# df_dM_dT = df_dM_dT.drop(dat_drop)
# df_dM_dT.reset_index(drop=True, inplace=True)
# # plt.plot(df_dM_dT['Temperature'] , df_dM_dT[0])

#%% calculate normalized derivitave weight loss via Rolling window OLS (%/degree)
window = 7
df_dnorM_dT = pd.DataFrame({'Temperature':IntigerT})
dT_min = int((window/2)-0.5) #calculates the number of indexies to shift data
# so that the value return is for the midle of the window
dT_max = Tmax-dT_min
dat_shift= list(range(0,dT_min)) # makes list of rows to delet later
# df_dM_dT = pd.DataFrame({'Temperature' : TRange[dT_min:dT_max]})
for x in range(Num_of_records):
    endog = df_percent_dry_weight[x] # this is the y values for  
    exog = sm.add_constant(IntigerT)
    # exog = sm.add_constant(IntigerT, prepend=False)
    mod = RollingOLS(endog, exog, window=window)
    fitted = mod.fit()
    temp = fitted.params
    temp=temp.reset_index(drop=True)
    temp=temp.drop(dat_shift) # drops first few values so that the window is for the center of the window instead of
    # an edge
    temp=temp.reset_index(drop=True) # resets index after deleting rows
    df_dnorM_dT[x] = temp['x1']
    
#code to drop first 3 and last 3 rows
last=df_dnorM_dT.last_valid_index()
allmost_last = last - dT_min
end_drop = list(range(allmost_last , last))
dat_drop = dat_shift + end_drop
df_dnorM_dT = df_dnorM_dT.drop(dat_drop)
df_dnorM_dT.reset_index(drop=True, inplace=True)
plt.plot(df_dnorM_dT['Temperature'] , df_dnorM_dT[0])

#%% Average deriviatve of weight loss (percent/degree)
No_T = df_dnorM_dT.iloc[ :, 1:33]
Mean = pd.DataFrame(df_dnorM_dT.loc[:,'Temperature'])
Mean['dM_dT'] = No_T.mean(axis = 1)
fig, ax = plt.subplots(figsize=(7.08, 6))
ax.plot(Mean['Temperature'] , Mean['dM_dT']*100)
ax.set_xlabel("Temperature (degrees C)", fontsize=9)
ax.set_ylabel("Derivitave Weight (d(M%)/dT)", fontsize=9)
# ax.set_ylim([-0.015,0])
# ax.set_xlim([200,300])

#HTML plot is here
# fig = px.line(x=Mean['Temperature'], y=Mean['dM_dT']*100)
# fig.write_html('output_data/figures/mean_derivitave_weight_loss.html', auto_open=True)
#%% Export data frame
#df_dnorM_dT has the change in mass per deagr (delta (fraction)/°C) for all of the "good" data that was
#imported in the TGA_Mass_loss_data_import
# df_dnorM_dT.to_csv("output_data/TGA_change_in_mass_per_degree_binned_1C_T.csv")
# pd.to_pickle(df_dnorM_dT,"pickle_jar/dMass_over_dT_bin_1C.pkl" )

#%% Second derivitave
window = 7
df_second_d = pd.DataFrame(Mean['Temperature'])
Center_of_window = int((window/2)-0.5) #calculates the number of indexies to shift data
# so that the value return is for the midle of the window
dT_max = Tmax-dT_min
dat_shift= list(range(0,Center_of_window)) # makes list of rows to delet later
# df_dM_dT = pd.DataFrame({'Temperature' : TRange[dT_min:dT_max]})
endog = Mean['dM_dT'] # this is the y values for 
exog = sm.add_constant(Mean['Temperature'])
# exog = sm.add_constant(IntigerT, prepend=False)
mod = RollingOLS(endog, exog, window=window)
fitted = mod.fit()
temp = fitted.params
temp=temp.rename(columns={"Temperature": "slope"})
temp=temp.reset_index(drop=True)
temp=temp.drop(dat_shift) # drops first few values so that the window is for the center of the window instead of
# an edge
temp=temp.reset_index(drop=True) # resets index after deleting rows
df_second_d['slope'] = temp['slope']

ColPal = ['#256676', '#1f0133', '#696fc4', '#9b1b5c']
fig, ax1 = plt.subplots(figsize=(7.08, 6))
ax1.plot(Mean['Temperature'] , Mean['dM_dT']*100,color=ColPal[1])
ax2=ax1.twinx()
ax2.plot(df_second_d['Temperature'] , df_second_d['slope'],color=ColPal[0])
ax2.tick_params(axis='y', labelsize=8, colors=ColPal[0])
ax1.set_ylabel("Derivitave Weight (% / °C)", fontsize=9, color=ColPal[1])
ax2.set_ylim([-4e-5, 4e-5])
ax2.set_ylabel("2nd Derivitave Weight", fontsize=9, color=ColPal[0])
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize=8, colors=ColPal[1])
ax1.set_xlabel("Temperature (°C)", fontsize=9)
fig.savefig('output_data/figures/mean_2nd_derivitave_weight_loss.svg', transparent=False, bbox_inches="tight")

#HTML plot is here
# fig = px.line(x=df_second_d['Temperature'], y=df_second_d['slope'])
# fig.write_html('output_data/figures/mean_2nd_derivitave_weight_loss.html', auto_open=True)

#%% Calculate weight loss over temperatures of interest

#calcuates the precentage of weight lost (normalized to dry weight at 105C) betwean 2 temperatures
def dryweightloss (T1,T2):
    loss = (df_percent_dry_weight.iloc[T1] - df_percent_dry_weight.iloc[T2])*100
    return loss
# index number for df_percent_dry_weight is °C and the weight loss is fraction of dry (105°C) weight 
fl_df2 = pd.DataFrame({'NEUP_Sample_#': Meta_df.loc[:,'NEUP_Sample_#']})
fl_df2['Al-OH'] = dryweightloss(180, 325)
fl_df2['Mg-OH'] = dryweightloss(325, 550)

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
#%% Split out data sets
# get list of file numbers for the data set that of samples activeate with
# 1ml and 0.5ml 10M NaOH per gram of cement precourser  aged for 24hrs at 35C before drying
# at time of progam is all of the good TGA data we have
Index_set_good = pd.read_csv('input_data/Sample_set_1ml_and_halfml_per_1gramPC.csv')

# creates df with the rows matching the index (file number) improrted in the above lines
Data_set_good2 = fl_df2.loc[Index_set_good['File number']]

#%% Export even more data frames

#Data set with all the 1mL o.5ml and Centroid v. NaOH concentration data
#For samples activated with NaOH
pd.to_pickle(Data_set_good2, "pickle_jar/TGA_data_good2.pkl") # wieght loss data

