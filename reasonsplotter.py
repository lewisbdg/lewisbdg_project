#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:58:49 2024

@author: lewisbolding
"""

import numpy as np
import matplotlib.pyplot as plt

clim = np.array([216, 259, float('nan'), 224, 234, 175, 146, 212, 
                 float('nan'), 237, 149, 16, float('nan'), 31, 66, 
                 269, 184, 215, 226, 115, 202, 250, 211])

years = np.arange(2001, 2024)

total = np.array([39006, 41111, float('nan'), 38995, 39888, 39837,
                  38681, 35167, float('nan'), 37540, 35075, 36488,
                  float('nan'), 35681, 36324, 35138, 34902, 32352,
                  31371, 29780, 27059, 28179, 25624])

prop = clim * 100 / total

plt.scatter(years, prop)

moved = np.zeros(len(ags)) 
moved_count = np.zeros(10)
for year in range(num_years): #Loop over all years
    for i in range(len(ags)): #Loop over all agents
        moved[i] = ags[i].history[year] != ags[i].history[year + 1] #Find which agents moved states for each year
    moved_count[year] = np.sum(moved) #Sum to find number of agents that moved in each year
    
fig, ax = plt.subplots() #Plot this data to see how migration numbers change each year
plt.plot(moved_count)
plt.xlabel("Year")
plt.ylabel("Number of moves")