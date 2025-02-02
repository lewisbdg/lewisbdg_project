#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:34:42 2025

@author: lewisbolding
"""

import matplotlib.pyplot as plt
import numpy as np



    
#fig, ax = plt.subplots() #Plot this data to see how migration numbers change each year
#plt.plot(moved_count)
#plt.xlabel("Year")
#plt.ylabel("Number of moves")

year_plot = 1999
year_idx = year_plot - 1990
state = "Texas"

num_child = 0
num_16 = 0
num_26 = 0
num_46 = 0
num_65 = 0

ags_in_state = []
for agidx in ags:
    if agidx.history[year_idx] == state:
       ags_in_state = np.append(ags_in_state, agidx)
       if agidx.age == "<16":
           num_child = num_child + 1
       elif agidx.age == "16-24":
           num_16 = num_16 + 1
       elif agidx.age == "25-44":
           num_26 = num_26 + 1
       elif agidx.age == "45-64":
           num_46 = num_46 + 1
       else:
           num_65 = num_65 + 1
         
num_child_mod = num_child * 100
num_16_mod = num_16 * 100
num_26_mod = num_26 * 100
num_46_mod = num_46 * 100
num_65_mod = num_65 * 100
        
state_data = data_1990s[data_1990s["1990"]["State"] == state]
num_child_act = np.sum(np.sum(state_data[state_data["1990"]["Age"] < 16][str(year_plot)]))
num_16_act = np.sum(np.sum(state_data[(state_data["1990"]["Age"] >= 16) & (state_data["1990"]["Age"] < 26)][str(year_plot)]))
num_26_act = np.sum(np.sum(state_data[(state_data["1990"]["Age"] >= 26) & (state_data["1990"]["Age"] < 46)][str(year_plot)]))
num_46_act = np.sum(np.sum(state_data[(state_data["1990"]["Age"] >= 46) & (state_data["1990"]["Age"] < 65)][str(year_plot)]))
num_65_act = np.sum(np.sum(state_data[state_data["1990"]["Age"] >= 65][str(year_plot)]))

X = [">16", "16-25", "26-45", "46-64", ">64"] 
actual = np.array([num_child_act, num_16_act, num_26_act, num_46_act, num_65_act])/1000000
model = np.array([num_child_mod, num_16_mod, num_26_mod, num_46_mod, num_65_mod])/1000000
  
X_axis = np.arange(len(X)) 
plt.bar(X_axis - 0.2, actual, 0.4, label = 'Data') 
plt.bar(X_axis + 0.2, model, 0.4, label = 'Model') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Age") 
plt.ylabel("Population / millions") 
plt.legend() 
plt.title("Modelled population distribution for " + state + " in 1999")
plt.show()
        

        