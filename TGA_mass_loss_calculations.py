# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:39:16 2022

@author: Titus
"""

#%% Intial setup
import os
import re
import pandas as pd
import numpy as np

# get varialble from pickle jar
Meta_df = pd.read_pickle("pickle_jar/Meta_df.pkl")

lst_TGA_data= pd.read_pickle("pickle_jar/TGA_ex_data.pkl")
#%%