# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:47:10 2022
Use this code to plot TGA data and best fitting curve
@author: Titus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  # keeps text as text in Illustrator
plt.rcParams['ps.fonttype'] = 42  # keeps text as text in Illustrator
plt.rcParams['font.sans-serif'] = "calibri"

# %% Import dat from pickle jar

TGA_data = pd.read_pickle("pickle_jar/TGA_data_good.pkl")  # weight loss data
Meta_data = pd.read_pickle("pickle_jar/Meta_set_good.pkl")
Mineral_data = pd.read_pickle("pickle_jar/Mineral_set_good.pkl")  # ICP results

# %% Make df with the data needed for subset selection
dat = pd.concat([Mineral_data, TGA_data], axis=1)
dat['Concentration(mM)'] = Meta_data.loc[:, 'Concentration(mM)']
dat['L/S'] = Meta_data.loc[:, 'L/S']
dat = dat.apply(pd.to_numeric)  # converts all values to numbers
# dat.to_csv("output_data/22_07_26data.csv")
# %% pick x and y data

Xdat = dat[['MgO']] * 100
Ydat = dat[['Al2O3']] * 100
Zdat = dat[['Tight_bound_water']] * 100  # change this to change your responce variable


# calculte the line of fit
def Zfit(MgO, Al2O3):
    return MgO * 0.2685 + Al2O3 * 0.2912


Xov = []
Yov = []
Zov = []

Xun = []
Yun = []
Zun = []

Zpredict = pd.DataFrame(Zfit(Xdat.loc[:, 'MgO'], Ydat.loc[:, 'Al2O3']))

# split over and under predictions for graphing
for i in range(len(Zdat)):
    if Zdat.iloc[i, 0] > Zpredict.iloc[i, 0]:
        Xov.append(Xdat.iloc[i, 0])
        Yov.append(Ydat.iloc[i, 0])
        Zov.append(Zdat.iloc[i, 0])
    else:
        Xun.append(Xdat.iloc[i, 0])
        Yun.append(Ydat.iloc[i, 0])
        Zun.append(Zdat.iloc[i, 0])

x = np.linspace(5, 50, 5)
y = np.linspace(5, 50, 5)
X, Y = np.meshgrid(x, y)
Z = Zfit(X, Y)
# %% 3D plot of fit


fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.contour3D(X, Y, Z, 150, cmap='gray', alpha=.5) #cmap='Wistia',
# plt.markers.MarkerStyle(marker='x')
ax.scatter3D(Xun, Yun, Zun, color='#001282')  # marker= '*', s=50

ax.plot_surface(X, Y, Z, color='gray', alpha=.3, edgecolor='black')

ax.scatter3D(Xov, Yov, Zov, color='#99004d')  # c=Zdat, cmap='winter')
# ax.plot3D(Xdat, Ydat, Zfit(Xdat,Ydat), 'gray')
ax.set_xlabel('MgO (wt. %)')
ax.set_ylabel('Al2O3 (wt. %)')
ax.set_zlabel('Weight Loss (%)')
ax.view_init(25, -70)

# # Uncomment this line to save the figure.
fig.savefig('output_data/figures/TightWater_curve.pdf', transparent=True, bbox_inches="tight")
plt.show(block=True)

# %% plot predicted vs observed

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(Zdat.iloc[:, 0], Zpredict.iloc[:, 0], 'x')

# create reference line along y = x to show the desired behavior
min_max = np.array([Zdat.min(), Zdat.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')

# fig.savefig('output_data/figures/TightWater_Observ_v_predict.pdf', transparent=True, 
#             bbox_inches="tight")
plt.show(block=True)
