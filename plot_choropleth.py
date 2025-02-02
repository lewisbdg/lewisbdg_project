#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:52:50 2025

@author: lewisbolding
"""
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

import plotly.graph_objects as go

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

import matplotlib.pyplot as plt
import numpy as np

x_out = np.zeros(51)
x_in = np.zeros(51)
for i in ags: #Loop over all agents
    if i.history[0] != i.history[9]:
        idx_state_1 = np.where(i.history[0] == list_states_grav)[0][0]
        idx_state_2 = np.where(i.history[9] == list_states_grav)[0][0]
        x_in[idx_state_1] = x_in[idx_state_1] + 1
        x_out[idx_state_2] = x_out[idx_state_2] + 1
    
x_in = x_in * 10000
x_out = x_out * 10000

x_in = np.delete(x_in, 8)
x_out = np.delete(x_out, 8)

x_net = x_in - x_out

fig = go.Figure(data=go.Choropleth(
    locations=df['code'], # Spatial coordinates
    z = x_net.astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = [[0, 'rgba(214, 39, 40, 0.85)'],   
               [0.88, 'rgba(255, 255, 255, 0.85)'],  
               [1, 'rgba(4,59,92, 0.85)']],
    colorbar_title = "Number of movers",
))

fig.update_layout(
    title_text = 'Net Internal US Migration in the 1990s',
    geo_scope='usa', # limite map scope to USA
)
fig.show()
fig.write_html("/Users/lewisbolding/project/x_net.html")