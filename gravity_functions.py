#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:53:38 2025

@author: lewisbolding
"""

#GRAVITY MODEL
import pandas as pd
import numpy as np

list_states_grav = list_states.name

G = 5 * 10 ** (-13)
P1_coeff = 0.927
P2_coeff = 1.769
dist_coeff = 0.797

def move_num(initial_state, final_state, year):
    P1 = total_pop(initial_state, year)[0]
    initial_state_idx = np.where(list_states_grav == initial_state)[0][0]
    P2 = total_pop(final_state, year)[0]
    final_state_idx = np.where(list_states_grav == final_state)[0][0]
    dist = state_dist[initial_state_idx, final_state_idx]
    num_moved = G * (P1 ** P1_coeff) * (P2 ** P2_coeff) / (dist ** dist_coeff)
    return num_moved
    
def generate_prob_array(year):
    moved_array = np.zeros([51, 51])
    for initial_state_idx_loop in range(len(list_states_grav)):
        initial_state_name = list_states_grav[initial_state_idx_loop]
        for final_state_idx_loop in range(len(list_states_grav)):
            final_state_name = list_states_grav[final_state_idx_loop]
            if initial_state_idx_loop == final_state_idx_loop:
                moved_array[initial_state_idx_loop, final_state_idx_loop] = 0
            else:
                moved_array[initial_state_idx_loop, final_state_idx_loop] = move_num(initial_state_name, 
                                                                                  final_state_name, year)
    
        total_moved = np.sum(moved_array[initial_state_idx_loop, :])
        total_stayed = total_pop(initial_state_name, year)[0] - total_moved
        moved_array[initial_state_idx_loop, initial_state_idx_loop] = total_stayed
    moved_array_cum = np.cumsum(moved_array, axis = 1)
        
    for initial_state_idx_loop in range(len(list_states_grav)):
        moved_array_cum[initial_state_idx_loop, :] = moved_array_cum[initial_state_idx_loop, :] / moved_array_cum[initial_state_idx_loop, -1]
    return moved_array_cum