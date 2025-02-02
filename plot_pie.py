#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:04:32 2025

@author: lewisbolding
"""

import matplotlib.pyplot as plt
import numpy as np

state = "Louisiana"

num_agents_1 = np.zeros(10)
num_agents_2 = np.zeros(10)
num_agents_3 = np.zeros(10)
num_agents_4 = np.zeros(10)
num_agents_5 = np.zeros(10)
state_data = data_1990s[data_1990s["1990"]["State"] == state]

for i in range(0, 10):
    year_plot = 1990 + i
    year_str = str(year_plot)
    new_array_1 = state_data[state_data["1990"]["Age"] < 16][year_str]
    new_array_2 = state_data[(state_data["1990"]["Age"] >= 16) & (state_data["1990"]["Age"] < 26)][year_str]
    new_array_3 = state_data[(state_data["1990"]["Age"] >= 26) & (state_data["1990"]["Age"] < 46)][year_str]
    new_array_4 = state_data[(state_data["1990"]["Age"] >= 46) & (state_data["1990"]["Age"] < 65)][year_str]
    new_array_5 = state_data[state_data["1990"]["Age"] >= 65][year_str]
    if i == 0:
        new_array_1 = new_array_1.drop(columns=["Age", "State"])
        new_array_2 = new_array_2.drop(columns=["Age", "State"])
        new_array_3 = new_array_3.drop(columns=["Age", "State"])
        new_array_4 = new_array_4.drop(columns=["Age", "State"])
        new_array_5 = new_array_5.drop(columns=["Age", "State"])
    num_agents_1[i] = np.sum(np.sum(new_array_1))
    num_agents_2[i] = np.sum(np.sum(new_array_2))
    num_agents_3[i] = np.sum(np.sum(new_array_3))
    num_agents_4[i] = np.sum(np.sum(new_array_4))
    num_agents_5[i] = np.sum(np.sum(new_array_5))

fig, ax = plt.subplots() #Plot this data to see how migration numbers change each year
plt.plot(num_agents_5)
plt.xlabel("Year")
plt.ylabel("Number of moves")

yX = 0
year_5 = np.array([num_agents_1[yX], num_agents_2[yX], num_agents_3[yX], num_agents_4[yX], num_agents_5[yX]])
labels_y5 = ["<16", "16-25", "26-45", "46-64", ">64"]

ax.pie(year_5, labels=labels_y5,
       autopct='%.2f')
       
# num_16_act = np.sum(np.sum(state_data[(state_data["1990"]["Age"] >= 16) & (state_data["1990"]["Age"] < 26)][str(year_plot)]))
# num_26_act = np.sum(np.sum(state_data[(state_data["1990"]["Age"] >= 26) & (state_data["1990"]["Age"] < 46)][str(year_plot)]))
# num_46_act = np.sum(np.sum(state_data[(state_data["1990"]["Age"] >= 46) & (state_data["1990"]["Age"] < 65)][str(year_plot)]))
# num_65_act = np.sum(np.sum(state_data[state_data["1990"]["Age"] >= 65][str(year_plot)]))

