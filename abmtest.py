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
rating between their origin and destination state, and distance. 
Interactions with other agents allows them to exchange information on each 
state's climate rating. This new information may then affect future decisions
migrate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

list_states = pd.read_csv("/Users/lewisbolding/Downloads/us-state-capitals.csv") #Read table of each state, its state capital, and lat/long coordinates

climate_rating = np.random.rand(50) * 2 - 1 #Generate random "climate ratings" between -1 and 1 for each state. These can be changed to represent real data later.
list_states["climate"] = climate_rating #Add additional climate score column to table.

def state_distance(state1, state2): #Function calculates distance between two state capitals using lat/long coordinates
    lat1 = np.deg2rad(list_states.loc[state1, "latitude"])
    long1 = np.deg2rad(list_states.loc[state1, "longitude"])
    lat2 = np.deg2rad(list_states.loc[state2, "latitude"])
    long2 = np.deg2rad(list_states.loc[state2, "longitude"])
    dist = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(long2-long1)) * 6371
    return dist

state_dist = np.zeros([50, 50]) #Generate empty state distance array
for state1 in range(0,50):
    for state2 in range(0, 50):
        if state1 == state2:
            state_dist[state1, state2] = 0 #Set distance to 0 for two identical states
        else:
            state_dist[state1, state2] = state_distance(state1, state2) #Calculate distance between two different states, save values in array
            
class agent: #Define agent class
    def __init__(self, origin_state, alarm_threshold, knowledge, state_history):
        self.state = origin_state #Initial state before any migration decisions are made
        self.alarm = alarm_threshold #Climate score below which agents choose to migrate
        self.knowledge = knowledge #Each agent's knowledge of climate score for each state. Agents know less about states that are further away
        self.history = state_history #Record of where agents move year-by-year

ags = list() #Create empty list of agents
for idx_state in range(50): #Loop over 50 states
    state = list_states.name[idx_state] #Find state name from table
    dist_array = state_dist[idx_state, :] #Find distance from chosen state to all other states
    max_dist = np.max(dist_array) #Find maximum distance
    uncertainty_array = 1 - dist_array / max_dist #Multipler that decreases with distance, to represent lack of knowledge of more distant states
    knowledge_array = climate_rating * uncertainty_array #Generate agent knowledge by multiplying climate score data by uncertainty array
    state_history = state #Generate state history array (appended later)
    for idx_agent in range(10): #Generate 10 agents for each state
        alarm = np.random.rand() * 2 - 1 #Generate climate score threshold between -1 and 1, below which agents will migrate
        ags.append(agent(state, alarm, knowledge_array, state_history)) #Add new agent to list
        

for year in range(10): #Simulate over 10 years

    for i in ags: #Loop over all agents
        state_idx = list_states.name == i.state #Find index corresponding to agent's current state
        state_clim = i.knowledge[state_idx].item() #Find climate score for agent's current state
        if state_clim < i.alarm: #If state climate score is below agent's threshold, choose to migrate
            gain = i.knowledge - state_clim #Calculate difference between all other states and agent's current state ("climate gain")
            gain_dist = np.nan_to_num(gain / state_dist[state_idx][0]) #Calculate gain per unit distance
            index_max = np.argmax(gain_dist) #Find index of state with maximum gain per unit distance
            new_state = list_states.name[index_max] #Find name of this state
            i.state = new_state #Change agent's current state to this new state - the agent has migrated!
            new_state_history = np.append(i.history, new_state) #Add this state to the agent's state history 
            i.history = new_state_history #Add updated state history to agent's profile
            i.knowledge[index_max] = list_states.climate[index_max] #Update agent's knowledge to include accurate data for new state
        else: #If state climate score is above agent's threshold, choose not to migrate
            new_state_history = np.append(i.history, i.state) #Update agent's state history - no change
            i.history = new_state_history

    num_interactions = 100 #Choose number of interactions between agents
    for idx_interaction in range(num_interactions): #Loop over all interactions
        a = np.random.randint(500) #Choose agent at random to interact. Agents can only interact with those in the same state
        match = np.zeros(500) #Generate empty state matching vector
        for i in range(len(ags)): #Loop over all agents
            match[i] = ags[i].state == ags[a].state #Find other agents in the same state as this agent
            #if len(np.where(match)[0]) <= 1:
                #exit()
        b = rd.choice(np.where(match)[0]) #Choose second agent to interact
        if a == b:
            pass #Skip interaction if the same agent is chosen twice
        else:
            knowledge_a = ags[a].knowledge
            knowledge_b = ags[b].knowledge
            knowledge_a_new = (knowledge_a + 0.1 * knowledge_b) / 1.1 #Agent B shares some knowledge with agent A
            knowledge_b_new = (0.1 * knowledge_a + knowledge_b) / 1.1 #Agent A shares some knowledge with agent B
            ags[a].knowledge = knowledge_a_new #Update agents' profiles with this new knowledge
            ags[b].knowledge = knowledge_b_new

##### DATA ANALYSIS #####

final_state_list = np.array([]) #Generate empty array of final states
for i in ags: #Loop over all agents
    final_state = i.history[10] #Find agent's final state
    final_state_list = np.append(final_state_list, final_state) #Update array with agent's final state

count = np.zeros(50) #Generate empty array of count of each state in the final state array
colour = np.array([]) #Generate empty colour array
pie_chart_labels = list_states.name #Create array of state names from table
for i in range(50): #Loop over all states
    match = final_state_list == pie_chart_labels[i] #Generate boolean list for agents present in this state at the end of the simulation
    count[i] = sum(match) #Find number of agents present in this state at the end of the simulation
    if list_states.climate[i] > 0.5: #Assign each state a colour based on its climate score
        colour = np.append(colour, 'darkgreen')
    elif list_states.climate[i] <= 0.5 and list_states.climate[i] > 0:
        colour = np.append(colour, 'lightgreen')
    elif list_states.climate[i] <= 0 and list_states.climate[i] > -0.5:
        colour= np.append(colour, 'tomato')
    else:
        colour = np.append(colour, 'darkred')

mask = count > 5 #Filter out states with fewer than 5 agents at end of simulation
new_pie_chart_labels = np.append(pie_chart_labels[mask], "Other") #Add "other" category to state labels
other_count = np.sum(count[mask == 0]) #Find count of agents in "other" states
new_count = np.append(count[mask], other_count) #Append filtered count array with "other" values
new_colour = np.append(colour[mask], 'yellow') #Append filtered colour array with yellow representing "other" states

fig, ax = plt.subplots() #Generate pie chart for final states
ax.pie(new_count, labels=new_pie_chart_labels,
       colors=new_colour, 
        textprops={'fontsize': 8})

moved = np.zeros(500) 
moved_count = np.zeros(10)
for year in range(9): #Loop over all years
    for i in range(500): #Loop over all agents
        moved[i] = ags[i].history[year] != ags[i].history[year + 1] #Find which agents moved states for each year
    moved_count[year] = np.sum(moved) #Sum to find number of agents that moved in each year
    
fig, ax = plt.subplots() #Plot this data to see how migration numbers change each year
plt.plot(moved_count)
plt.xlabel("Year")
plt.ylabel("Number of moves")
    
        


        
