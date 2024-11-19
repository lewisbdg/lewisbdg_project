#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:45:19 2024

@author: lewisbolding

This simple script models interactions between agents in determining
migration flows across 50 states. Each state has a "climate rating" between
1 and -1, with -1 representing the greatest danger. Agents know exactly the
rating of their own state, with accuracy decreasing with distance. When
deciding on migration destination, agents consider the difference in climate
rating between their origin and destination state, distance, and their own
personal willingness to migrate. Interactions with other agents allows
them to exchange information on each state's climate rating.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

list_states = pd.read_csv("/Users/lewisbolding/Downloads/us-state-capitals.csv")

climate_rating = np.random.rand(50)
list_states["climate"] = climate_rating

def state_distance(state1, state2):
    lat1 = np.deg2rad(list_states.loc[state1, "latitude"])
    long1 = np.deg2rad(list_states.loc[state1, "longitude"])
    lat2 = np.deg2rad(list_states.loc[state2, "latitude"])
    long2 = np.deg2rad(list_states.loc[state2, "longitude"])
    dist = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(long2-long1)) * 6371
    return dist

state_dist = np.zeros([50, 50])
for state1 in range(0,50):
    for state2 in range(0, 50):
        if state1 == state2:
            state_dist[state1, state2] = 0
        else:
            state_dist[state1, state2] = state_distance(state1, state2)
            
class agent:
    def __init__(self, origin_state, alarm_threshold, knowledge):
        self.state = origin_state
        self.alarm = alarm_threshold
        self.knowledge = knowledge
        
    def __str__(self):
        return f"{self.state}({self.alarm})"

ags = list()
for idx_state in range(50):
    state = list_states.name[idx_state]
    dist_array = state_dist[idx_state, :]
    max_dist = np.max(dist_array)
    uncertainty_array = 1 - dist_array / max_dist 
    knowledge_array = climate_rating * uncertainty_array
    for idx_agent in range(10):
        alarm = np.random.rand()
        ags.append(agent(state, alarm, knowledge_array))
        
for i in ags:
    state_idx = list_states.name == i.state
    state_clim = i.knowledge[state_idx].item()
    if state_clim < i.alarm:
        gain = i.knowledge - state_clim
        gain_dist = np.nan_to_num(gain / state_dist[state_idx][0])
        index_max = np.argmax(gain_dist)
        new_state = list_states.name[index_max]
        i.state = new_state
        i.knowledge[index_max] = list_states.climate[index_max]
        
a = np.random.randint(500)
match = np.zeros(500)
for i in range(len(ags)):
    match[i] = ags[i].state == ags[a].state
if len(np.where(match)[0]) <= 1:
    quit()
b = rd.choice(np.where(match)[0])
while a == b:
    b = rd.choice(np.where(match)[0])
    





        
    
        


        
