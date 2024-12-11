#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:48:39 2024

@author: lewisbolding
"""

import pandas as pd
import numpy as np

colnames = ["ID", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df = pd.read_csv("/Users/lewisbolding/Downloads/climdiv-phdist-v1.0.0-20241205.txt", sep='\s+', names = colnames, converters={'ID': str})
df = df.drop(df.index[6240:])

clim_state_idxs = np.array(["001", "002","003","004","005","006","007","008","009","010",
                   "011", "012","013","014","015","016","017","018","019","020",
                   "021", "022","023","024","025","026","027","028","029","030",
                   "031", "032","033","034","035","036","037","038","039","040",
                   "041", "042","043","044","045","046","047","048"])
clim_state_names = np.array(['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'])

ID_states_names = np.array([])
ID_states_years = np.array([])
for i in range(len(df)):
    name_state = clim_state_names[df.ID[i][0:3] == clim_state_idxs][0]
    ID_states_names = np.append(ID_states_names, name_state)
    year = df.ID[i][6:10]
    ID_states_years = np.append(ID_states_years, year)
    
df["State"] = ID_states_names
df["Year"] = ID_states_years

cols = df.columns.tolist()
cols = cols[-2:] + cols[:-2]
df = df[cols].drop(columns="ID")
df["Mean"] = df.iloc[:, -12:-1].sum(axis=1)/12

DC_data = df.loc[2210:2339]
DC_data.State = "District of Columbia"
DC_data.index = DC_data.index - 1170

HI_data = df.loc[390:519]
HI_data.State = "Hawaii"
HI_data.index = HI_data.index + 1040

AK_data = df.loc[5720:5849]
AK_data.State = "Alaska"
AK_data.index = AK_data.index - 5590

data_a = df[:130]

data_b = df[130:910]
data_b.index = data_b.index + 130

data_c = df[910:1170]
data_c.index = data_c.index + 260

data_d = df[1170:]
data_d.index = data_d.index + 390

rainfall = pd.concat([data_a, AK_data, data_b, DC_data, data_c, HI_data, data_d])

