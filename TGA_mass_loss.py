# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:43:30 2022

@author: Titus
"""
# Used to import TGA and MS data from excell files exported from TA Trios Software
#%% Intial setup
import os
import re
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
NameMeta_df = pd.DataFrame({'File#': A_fnumb})
NameMeta_df['PC_Name'] = A_PC_Name
NameMeta_df['Concentration(mM)'] = A_Con
NameMeta_df['L/S'] = A_LS   
print(NameMeta_df)

#%%Read in the data from excell
Pan_Num = []
s_Mass = []
ex_Mass = []
NEUP_num = []
x = 0
tempMeta = pd.read_excel(os.path.join(datpath, TGA_names[x]), sheet_name=0, header=None)
print(tempMeta[tempMeta[0]=='Pan Number'].index.values)


#%%temp break
for x in TGA_count:
    #Read the TGA data
    tempTGA = pd.read_excel(os.path.join(datpath, TGA_names[x]), sheet_name=1, header=[1,2])
    TGA_listOdf.append(tempTGA)
    #Read the TGA meta data
    # tempMS_H20 = pd.read_excel(os.path.join(datpath, MS_names[x]), sheet_name="18.0 AMU", header=[1,2])
    # MS_H2O_listOdf.append(tempMS_H20)
    # tempMS_C02 = pd.read_excel(os.path.join(datpath, MS_names[x]), sheet_name="44.0 AMU", header=[1,2])
    # MS_CO2_listOdf.append(tempMS_C02) 
   
#%% Ready to polot ?

#Values for setting that are used multple places
ColPal = ['#256676', '#1f0133', '#696fc4', '#9b1b5c']
lnthikness= 1
legspot = 'upper right' # Determines where legend is placed
Wt_loss_dat = []
Wt_loss_dat2 = []
def Plot_TGA_MS(index):
    font = FontProperties()
    font.set_family('sans-serf')
    font.set_name('Arial')
    font.set_size(9)

    #Get TGA data out of dataframe
    df = TGA_listOdf[index]
    TGA_T = np.array(df.loc[:,'Temperature'])
    TGA_M = np.array(df.loc[:,'Weight'])
    TGA_dM = np.array(df.loc[:,'Deriv. Weight'])

    
    # find index of first time T reaches 30C
    inx30 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 30 )
    # find index of first time T reaches 105C
    inx105 = next(x for x, val in enumerate(TGA_T)
                                  if val >= 105 )
    
    print(Graph_Title[index])
    print("index at 30C ", inx30)
    print("mass percent at 30C ", TGA_M[inx30])
    norFact= 100 / TGA_M[inx30]
    print(norFact)
    wtloss = 100 - TGA_M[-1] * norFact
    print("weight loss form 30 to 1000C", wtloss, "%")
    Wt_loss_dat.append(float(wtloss))
    stru_wt_loss = ( TGA_M[inx105] - TGA_M[-1]) * norFact
    Wt_loss_dat2.append(float(stru_wt_loss))

    # #Get MS Data
    df = MS_CO2_listOdf[index]
    CO2_sig = np.array(df.loc[:,'Ion Current'])
    CO2_T = np.array(df.loc[:,'Temperature'])
    df = MS_H2O_listOdf[index]
    H2O_sig = np.array(df.loc[:,'Ion Current'])
    H2O_T = np.array(df.loc[:,'Temperature'])

    #Plotting
    fig = plt.figure(figsize=[7.08, 6] ,constrained_layout=True)
    gs = fig.add_gridspec(4, 1)

    ax1 = fig.add_subplot(gs[:-1, :])
    ax1.set_title(Graph_Title[index])
    ax1.plot(TGA_T,TGA_M * norFact, linewidth=lnthikness, color=ColPal[1])    

    ax1.set_ylabel("Weight (%)", fontsize=9, color=ColPal[1])
    ax1.tick_params(axis='x', labelsize=8)
    ax1.xaxis.set_major_locator(MultipleLocator(200))
    ax1.xaxis.set_minor_locator(MultipleLocator(50))
    ax1.tick_params(axis='y', labelsize=8, colors=ColPal[1])
    ax1.set_xlim([30,1000])
    ax1.set_ylim([60,100])

    # Add second y-axis and plot dM
    ax2=ax1.twinx()
    ax2.plot(TGA_T,TGA_dM, linewidth=lnthikness, color=ColPal[0])
    ax2.tick_params(axis='y', labelsize=8, colors=ColPal[0])
    ax2.set_ylabel("Derivitave Weight (% / °C)", fontsize=9, color=ColPal[0])
    ax2.set_ylim([-.3,.3])
    # ax1.legend(["Weight"],loc='upper right')
    # ax2.legend(["derivative"],loc='upper right')

    #Add second plot with MS data
    ax3 = fig.add_subplot(gs[-1, :]) 
    ax3.plot(H2O_T, H2O_sig , linewidth=lnthikness, color=ColPal[2],)
    ax3.plot(CO2_T, CO2_sig , linewidth=lnthikness, color=ColPal[3])
    ax3.set_xlim([30,1000])
    ax3.set_xlabel("Temperature (°C)", fontsize=9)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.xaxis.set_major_locator(MultipleLocator(100))
    ax3.xaxis.set_minor_locator(MultipleLocator(25))
    ax3.tick_params(axis='y', labelsize=8)
    ax3.set_ylabel("Ion Current (mA)", fontsize=9)
    ax3.legend(["18 AMU ($H_2O$)", "44 AMU ($CO_2 $)"],loc='upper right', fontsize=9)
    
    svg_name_path = 'Plots/'+ Graph_Title[index] + '.svg'
    # # Uncomment this line to save the figure.
    # fig.savefig(svg_name_path, transparent=False, bbox_inches="tight")
    return fig
#%%

for x in TGA_MS_count:
    Plot_TGA_MS(x)
    
wt_loss_dictionary = {"File" : TGA_names, "Name" : Graph_Title, "Weight Loss 30-1000C (%)" : Wt_loss_dat
, "Weight Loss 105-1000C (%)" : Wt_loss_dat2}
wt_loss_df = pd.DataFrame(wt_loss_dictionary)
wt_loss_df.to_csv('Plots/weight_loss_table.csv', index=False)
