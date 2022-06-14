# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:43:30 2022

@author: Titus
"""
# Used to import TGA and MS data from excell files exported from TA Trios Software
#%% Intial setup
import os
import re
import pickle
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
# from matplotlib.font_manager import FontProperties

#%% Import data

datpath = 'input_data' # directory where data is stored relative to py script location
f_name = (os.listdir(datpath))#list of files in the directory of datpath

file_count = range(len(f_name))
TGA_names = []

# Makes a list of TGA file and a seprate list of MS files
for x in file_count:
    if f_name[x].startswith('TGA'):
        TGA_names.append(f_name[x])


TGA_count = range(len(TGA_names))
TGA_listOdf = []

#%%get sample info from name

A_fnumb = []
A_PC_Name = []
A_Con = []
A_LS = []

for x in TGA_count:
    #find the end of the PC name by finding first number
    m = re.search(r"\d", TGA_names[x])
    temp_name = TGA_names[x][4 : m.span()[1]-1]
    temp_name = temp_name.replace("Axial", "-Axial")
    temp_name = temp_name.replace("Corner", "-Corner")
    A_PC_Name.append(temp_name)
    #find the concentration by searching for M_
    n = re.search(r"M_", TGA_names[x])
    A_Con.append(TGA_names[x][n.span()[1]-6 : n.span()[1]-2])
    o = re.search(r"_LS", TGA_names[x])
    A_LS.append(TGA_names[x][o.span()[1]: o.span()[1]+4])
    A_fnumb.append(x)

# Place list of info into data frame    
Meta_df = pd.DataFrame({'File#': A_fnumb})
Meta_df['PC_Name'] = A_PC_Name
Meta_df['Concentration(mM)'] = A_Con
Meta_df['L/S'] = A_LS   
print(Meta_df)

#%%Read in the meta data from excell
Pan_Num = []
s_Mass = []
ex_Mass = []
NEUP_num = []

for x in TGA_count:
    tempMeta = pd.read_excel(os.path.join(datpath, TGA_names[x]), sheet_name=0,\
                         header=None)
    # find  pan number & add to list
    lt1 = list(tempMeta.iloc[tempMeta[tempMeta[0]=='Pan Number'].index.values, 1])
    Pan_Num.append(lt1[0])
    # starting mass of the sample as measured by TGA instrument and add to list
    lt2 = list(tempMeta.iloc[tempMeta[tempMeta[0]=='Sample Mass'].index.values, 1])
    lt2 = re.search("\d+\.\d+",lt2[0])
    s_Mass.append(lt2[0])
    # starting mass of the sample as measured by operatore on scale in room
    lt3 = list(tempMeta.iloc[tempMeta[tempMeta[0]=='mass (mg) external scale'].\
                             index.values, 1])
    ex_Mass.append(lt3[0])
    # find inhouse sample number and store to list
    lt4 = list(tempMeta.iloc[tempMeta[tempMeta[0]=='NEUP sample #'].
                              index.values, 1])
    NEUP_num.append(lt4[0])

Meta_df['Pan_#'] = Pan_Num
Meta_df['Initial_mass_TGA_mg'] = s_Mass 
Meta_df['Intial_mass_external_mg'] = ex_Mass
Meta_df['NEUP_Sample_#'] = NEUP_num
print(Meta_df)

#%% read expermantal data from excel
lst_o_data = []
for x in TGA_count:
    #Read the TGA expermental data into dataframe lst_o_data
    tempTGA = pd.read_excel(os.path.join(datpath, TGA_names[x]), sheet_name=1, header=[1,2])
    lst_o_data.append(tempTGA)
    
#%% Pickle data for use later
# creates file for pickle w = creat for writing, b = treat as binary file

pd.to_pickle(Meta_df,"pickle_jar/Meta_df.pkl" )
pd.to_pickle(lst_o_data,"pickle_jar/TGA_ex_data.pkl" )
# with open ('output_data/TGA_meta_data.pickle', 'wb') as f: 
#     pickle.dump('Meta_df', f)  

# with open ('output_data/TGA_ex_data.pickle', 'wb') as f: 
#     pickle.dump('lst_o_data', f)

#%% Output to excell

Meta_df.to_csv("output_data/TGA_Meta_Data.csv")

# for x in TGA_MS_count:
#     Plot_TGA_MS(x)
    
# wt_loss_dictionary = {"File" : TGA_names, "Name" : Graph_Title, "Weight Loss 30-1000C (%)" : Wt_loss_dat
# , "Weight Loss 105-1000C (%)" : Wt_loss_dat2}
# wt_loss_df = pd.DataFrame(wt_loss_dictionary)
# wt_loss_df.to_csv('Plots/weight_loss_table.csv', index=False)
