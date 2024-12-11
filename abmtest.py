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

num_years = 10
num_states = 51
agent_types = ['NH-W-M', 'NH-W-F', 'NH-B-M', 'NH-B-F','NH-AIAN-M', 'NH-AIAN-F', 'NH-API-M', 'NH-API-F', 'H-W-M', 'H-W-F', 'H-B-M', 'H-B-F', 'H-AIAN-M', 'H-AIAN-F', 'H-API-M', 'H-API-F']

list_states = pd.read_csv("/Users/lewisbolding/Downloads/us-state-capitals.csv") #Read table of each state, its state capital, and lat/long coordinates


line = pd.DataFrame({"name": "District of Columbia", "description": "District of Columbia", "latitude":38.8898, "longitude":-77.0091}, index=[9])
list_states = pd.concat([list_states.iloc[:8], line, list_states.iloc[8:]]).reset_index(drop=True)

def state_distance(state1, state2): #Function calculates distance between two state capitals using lat/long coordinates
    lat1 = np.deg2rad(list_states.loc[state1, "latitude"])
    long1 = np.deg2rad(list_states.loc[state1, "longitude"])
    lat2 = np.deg2rad(list_states.loc[state2, "latitude"])
    long2 = np.deg2rad(list_states.loc[state2, "longitude"])
    dist = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(long2-long1)) * 6371
    return dist

state_dist = np.zeros([51, 51]) #Generate empty state distance array
for state1 in list_states.index:
    for state2 in list_states.index:
        if state1 == state2:
            state_dist[state1, state2] = 0 #Set distance to 0 for two identical states
        else:
            state_dist[state1, state2] = state_distance(state1, state2) #Calculate distance between two different states, save values in array
            
class agent: #Define agent class
    def __init__(self, origin_state, alarm_threshold, state_history):
        self.state = origin_state #Initial state before any migration decisions are made
        self.alarm = alarm_threshold #Climate score below which agents choose to migrate
        self.history = state_history #Record of where agents move year-by-year

ags = list() #Create empty list of agents
for idx_state in list_states.index: #Loop over 50 states
    state = list_states.name[idx_state] #Find state name from table
    print(state)
    dist_array = state_dist[idx_state, :] #Find distance from chosen state to all other states
    max_dist = np.max(dist_array) #Find maximum distance
    state_history = state #Generate state history array (appended later)
    for type in agent_types:
        for age in range(0, 86):
            for idx_agent in range(int(num_agents(state, "1990", type, age))): #Generate 10 agents for each state
                alarm = np.random.rand() * 2 + 1 #Generate climate score threshold between -1 and 1, below which agents will migrate
                ags.append(agent(state, alarm, state_history)) #Add new agent to list
        

for year in range(num_years): #Simulate over 10 years
    year_actual = str(1980 + year)

    for i in ags: #Loop over all agents
        print(i.state)
        state_idx = state_idx = list_states.name == i.state
        state_idx2 = np.where((rainfall.State == i.state) & (rainfall.Year == year_actual))[0][0] #Find index corresponding to agent's current state
        state_clim = rainfall.Mean[state_idx2] #Find climate score for agent's current state
        if np.abs(state_clim) > i.alarm: #If state climate score is below agent's threshold, choose to migrate
            gain = rainfall.Mean[(np.where(rainfall.Year == year_actual))[0]] - state_clim #Calculate difference between all other states and agent's current state ("climate gain")
            gain_dist = np.nan_to_num(gain / state_dist[state_idx][0]) #Calculate gain per unit distance
            index_max = np.argmax(gain_dist) #Find index of state with maximum gain per unit distance
            new_state = list_states.name[index_max] #Find name of this state
            i.state = new_state #Change agent's current state to this new state - the agent has migrated!
            new_state_history = np.append(i.history, new_state) #Add this state to the agent's state history 
            i.history = new_state_history #Add updated state history to agent's profile
        else: #If state climate score is above agent's threshold, choose not to migrate
            new_state_history = np.append(i.history, i.state) #Update agent's state history - no change
            i.history = new_state_history



    num_interactions = 100 #Choose number of interactions between agents
    for idx_interaction in range(num_interactions): #Loop over all interactions
        a = np.random.randint(len(ags)) #Choose agent at random to interact. Agents can only interact with those in the same state
        match = np.zeros(len(ags)) #Generate empty state matching vector
        for i in range(len(ags)): #Loop over all agents
            match[i] = ags[i].state == ags[a].state #Find other agents in the same state as this agent
            #if len(np.where(match)[0]) <= 1:
                #exit()
        b = rd.choice(np.where(match)[0]) #Choose second agent to interact
        if a == b:
            pass #Skip interaction if the same agent is chosen twice
        else:
            pass
##### DATA ANALYSIS #####


final_state_list = np.array([]) #Generate empty array of final states
a = 0
for i in ags: #Loop over all agents
    a = a + 1
    if a%100 == 0:
        print(a)
    final_state = i.history[num_years] #Find agent's final state
    final_state_list = np.append(final_state_list, final_state) #Update array with agent's final state


#count = np.zeros(num_states) #Generate empty array of count of each state in the final state array
#colour = np.array([]) #Generate empty colour array
#pie_chart_labels = list_states.name #Create array of state names from table
#for i in range(num_states): #Loop over all states
    #match = final_state_list == pie_chart_labels[i] #Generate boolean list for agents present in this state at the end of the simulation
    #count[i] = sum(match) #Find number of agents present in this state at the end of the simulation
    #if list_states.climate[i] > 0.5: #Assign each state a colour based on its climate score
        #colour = np.append(colour, 'darkgreen')
    #elif list_states.climate[i] <= 0.5 and list_states.climate[i] > 0:
        #colour = np.append(colour, 'lightgreen')
    #elif list_states.climate[i] <= 0 and list_states.climate[i] > -0.5:
        #colour= np.append(colour, 'tomato')
    #else:
        #colour = np.append(colour, 'darkred')

#mask = count > 5 #Filter out states with fewer than 5 agents at end of simulation
#new_pie_chart_labels = np.append(pie_chart_labels[mask], "Other") #Add "other" category to state labels
#new_count = np.append(count[mask], other_count) #Append filtered count array with "other" values
#other_count = np.sum(count[mask == 0]) #Find count of agents in "other" states
#new_colour = np.append(colour[mask], 'yellow') #Append filtered colour array with yellow representing "other" states

#ax.pie(new_count, labels=new_pie_chart_labels,
       #colors=new_colour, 
       #textprops={'fontsize': 8})
       
#moved = np.zeros(len(ags)) 
#moved_count = np.zeros(10)
#for year in range(num_years): #Loop over all years
    #for i in range(len(ags)): #Loop over all agents
        #moved[i] = ags[i].history[year] != ags[i].history[year + 1] #Find which agents moved states for each year
    #moved_count[year] = np.sum(moved) #Sum to find number of agents that moved in each year
    
#fig, ax = plt.subplots() #Plot this data to see how migration numbers change each year
#plt.plot(moved_count)
#plt.xlabel("Year")
#plt.ylabel("Number of moves")
    



        
