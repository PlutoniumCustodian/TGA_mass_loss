# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:34:26 2022

@author: Titus
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#%% Import dat from pickle jar

TGA_data = pd.read_pickle("pickle_jar/TGA_data_good.pkl") # weight loss data
Meta_data = pd.read_pickle("pickle_jar/Meta_set_good.pkl")
Mineral_data = pd.read_pickle("pickle_jar/Mineral_set_good.pkl") #ICP results

#%% Make df with the data needed for making predictive model
dat = pd.concat([Mineral_data, TGA_data], axis=1)
dat['Concentration(mM)'] = Meta_data.loc[:,'Concentration(mM)']
dat['L/S'] = Meta_data.loc[:,'L/S']
dat = dat.apply(pd.to_numeric) #converts all values to numbers
dat= dat.reset_index() #re-sets index so polynomial matrix indexes will match y

#%% pick x and y data (Unbound_Water)

linX= dat[['Concentration(mM)', 'L/S']]

poly = PolynomialFeatures(degree=2) # calculates all 2nd order options of X
poly = poly.fit(linX)
polyX = poly.transform(linX)
X_options = pd.DataFrame(data=polyX, columns=poly.get_feature_names_out())
print('Names of x options', X_options.columns.values.tolist())

X = X_options.loc[:,['Concentration(mM) L/S', 'L/S^2']]
y = dat.loc[:,'Unbound_water']


#%% split data and train model
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
# fit on the transformed features!
# model = linear_model.LinearRegression().fit(Xptrain, ytrain) # linear was prone to overfitting
model = linear_model.Lasso(alpha=0.01).fit(Xtrain, ytrain)
print('Shape of X:    ', model.predict(Xtrain).shape)
print('Shape of y     ', ytrain.shape)

#%% plot the results
fig, ax = plt.subplots(figsize=(5, 5))
#ytrain = ytrain.to_numpy()
ax.plot(ytrain, model.predict(Xtrain), 'x', label='Training Data')
ax.plot(ytest, model.predict(Xtest), '.', label='Testing Data')

# create reference line along y = x to show the desired behavior
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.legend()
y_pred = model.predict(Xtrain)
residuals = y_pred - ytrain

r2 = 1 - np.var(residuals) / np.var(ytrain - ytrain.mean())

rmse = np.sqrt(np.mean(residuals**2))

print(f'train: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

y_pred = model.predict(Xtest)
residuals = y_pred - ytest
r2 = 1 - np.var(residuals) / np.var(ytest - ytest.mean())
rmse = np.sqrt(np.mean(residuals**2))

print(f'test:  Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

#%% pick x and y data (loose_bound_water)

linX= dat[['Concentration(mM)', 'L/S']]

poly = PolynomialFeatures(degree=2) # calculates all 2nd order options of X
poly = poly.fit(linX)
polyX = poly.transform(linX)
X_options = pd.DataFrame(data=polyX, columns=poly.get_feature_names_out())
print('Names of x options', X_options.columns.values.tolist())

X = X_options.loc[:,['L/S', 'Concentration(mM)^2']]
y = dat.loc[:,'loose_bound_water']


#%% split data and train model
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
# fit on the transformed features!
# model = linear_model.LinearRegression().fit(Xptrain, ytrain) # linear was prone to overfitting
model = linear_model.Lasso(alpha=0.01).fit(Xtrain, ytrain)
print('Shape of X:    ', model.predict(Xtrain).shape)
print('Shape of y     ', ytrain.shape)

#%% plot the results
fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(ytrain, model.predict(Xtrain), 'x', label='Training Data')
ax.plot(ytest, model.predict(Xtest), '.', label='Testing Data')

# create reference line along y = x to show the desired behavior
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.legend()
y_pred = model.predict(Xtrain)
residuals = y_pred - ytrain

r2 = 1 - np.var(residuals) / np.var(ytrain - ytrain.mean())

rmse = np.sqrt(np.mean(residuals**2))

print(f'train: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

y_pred = model.predict(Xtest)
residuals = y_pred - ytest
r2 = 1 - np.var(residuals) / np.var(ytest - ytest.mean())
rmse = np.sqrt(np.mean(residuals**2))

print(f'test:  Rsq = {r2:.3f}, RMSE = {rmse:.3f}')
