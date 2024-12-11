#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:18:59 2024

@author: lewisbolding
"""

#import pandas as pd
#data = pd.read_csv("/Users/lewisbolding/Downloads/estat_migr_emi3nxt.tsv", sep='\t|,')
#data.rename({'geo\\TIME_PERIOD': 'geo'}, axis=1, inplace=True)

#import pandas as pd
#data = pd.read_csv("/Users/lewisbolding/Downloads/estat_migr_imm5prv.tsv", sep='\t|,')
#data.rename({'geo\\TIME_PERIOD': 'geo'}, axis=1, inplace=True)

import pandas as pd
import numpy as np

colnames = ['Year', 'State', 'Age', 'NH-W-M', 'NH-W-F', 'NH-B-M', 'NH-B-F','NH-AIAN-M', 'NH-AIAN-F', 'NH-API-M', 'NH-API-F', 'H-W-M', 'H-W-F', 'H-B-M', 'H-B-F', 'H-AIAN-M', 'H-AIAN-F', 'H-API-M', 'H-API-F']
years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999']
state_idxs1 = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]
state_idxs = [float(i) for i in state_idxs1]
state_names = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

data_90 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh90.txt"), columns = colnames)
data_90 = data_90.drop(columns='Year')
data_90['State'] = data_90['State'].replace(state_idxs, state_names)

data_91 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh91.txt"), columns = colnames)
data_91 = data_91.drop(columns='Year')

data_92 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh92.txt"), columns = colnames)
data_92 = data_92.drop(columns='Year')

data_93 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh93.txt"), columns = colnames)
data_93 = data_93.drop(columns='Year')

data_94 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh94.txt"), columns = colnames)
data_94 = data_94.drop(columns='Year')

data_95 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh95.txt"), columns = colnames)
data_95 = data_95.drop(columns='Year')

data_96 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh96.txt"), columns = colnames)
data_96 = data_96.drop(columns='Year')

data_97 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh97.txt"), columns = colnames)
data_97 = data_97.drop(columns='Year')

data_98 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh98.txt"), columns = colnames)
data_98 = data_98.drop(columns='Year')

data_99 = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh98.txt"), columns = ['Year', 'State', 'Age', 'NH-W-M', 'NH-W-F', 'NH-B-M', 'NH-B-F','NH-AIAN-M', 'NH-AIAN-F', 'NH-API-M', 'NH-API-F', 'H-W-M', 'H-W-F', 'H-B-M', 'H-B-F', 'H-AIAN-M', 'H-AIAN-F', 'H-API-M', 'H-API-F'])
data_99 = data_99.drop(columns='Year')

data_1990s = pd.concat([data_90, data_91, data_92, data_93, data_94, data_95, data_96, data_97, data_98, data_99], axis=1)
data_1990s.columns = pd.MultiIndex.from_product([years, np.delete(colnames, 0)], 
                                        names=['Year', 'Demographics'])


for yearidx in range(1, 9):
    year = years[yearidx]
    data_1990s = data_1990s.drop(columns = (year, 'Age'))
    data_1990s = data_1990s.drop(columns = (year, 'State'))
    
def total_pop(state, year):
    stateidxs = data_1990s["1990"].State == state
    usefuldata = data_1990s[year][stateidxs]
    if year == "1990":
        usefuldata = usefuldata.drop(columns=['Age', 'State'])
    tot = usefuldata.sum().sum()
    return tot, usefuldata

def num_agents(state, year, agent_type, age):
    tot = total_pop(state, year)[0]
    data = total_pop(state, year)[1]
    num_type = np.array(data[agent_type])[0]
    #prop = round((num_type / tot) * 10000)
    return round(num_type / 10000)
