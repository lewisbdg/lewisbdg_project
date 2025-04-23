

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import time
import copy
import plotly.io as pio
import random as rd
pio.renderers.default='browser'

# %% GLOBAL VARIABLES

fidelity = 1000 #Number of individuals per agent
num_years = 35 #Number of years for model
seed = 450 #Seed for migration decisions
np.random.seed(seed)

global_temp = np.array([0.33, 0.33, 0.33, 0.33, 0.34, 0.36, 0.40, 0.42, 0.44, 0.47, 0.5, 0.52, #Global temperature
                        0.54, 0.58, 0.61, 0.62, 0.62, 0.63, 0.64, 0.64, 0.64, 0.66, 0.69, 0.74,
                        0.78, 0.83, 0.87, 0.91, 0.93, 0.94, 0.97, 1.03, 1.08, 1.14, 1.20]) 

RCP26 = np.append(global_temp[:-1], np.linspace(1.20, 1.7, 75)) #Temperature for RCP pathways
RCP45 = np.append(global_temp[:-1], np.linspace(1.20, 2.5, 75))
RCP85 = np.append(global_temp[:-1], np.linspace(1.20, 4.3, 75))

projection = RCP45 #Choose pathway for current simulation
str_proj = "RCP45"

# %% POPULATION DATA AND FUNCTIONS

list_states = pd.read_csv("/Users/lewisbolding/Downloads/us-state-capitals.csv") #List of state capitals with coordinates
line = pd.DataFrame({"name": "District of Columbia", "description": "District of Columbia", "latitude":38.8898, "longitude":-77.0091}, index=[9])
list_states = pd.concat([list_states.iloc[:8], line, list_states.iloc[8:]]).reset_index(drop=True)
list_states = pd.concat([list_states.iloc[:9], list_states.iloc[10:12], list_states.iloc[[9]], list_states.iloc[12:32], list_states.iloc[[34]], list_states.iloc[32:34], list_states.iloc[35:]]).reset_index(drop=True)

list_cities = pd.read_excel("/Users/lewisbolding/project/uscities.xlsx") #List of all US cities with coordinates
reduced_list_cities = (list_cities.loc[list_cities.population>=50000]).iloc[:, [0, 3, 6, 7]]
reduced_list_cities = reduced_list_cities[reduced_list_cities.state_name != "Puerto Rico"] #Remove Puerto Rico from dataset

colnames = ['Year', 'State', 'Age', 'NH-W-M', 'NH-W-F', 'NH-B-M', 'NH-B-F','NH-AIAN-M', 'NH-AIAN-F', 'NH-API-M', 'NH-API-F', 'H-W-M', 'H-W-F', 'H-B-M', 'H-B-F', 'H-AIAN-M', 'H-AIAN-F', 'H-API-M', 'H-API-F']
state_idxs_for_file = np.delete(np.arange(1, 57).astype(float), [2, 6, 13, 42, 51]) #Removing unimportant data
state_names = list_states.name
data_90_old = pd.DataFrame(np.loadtxt("/Users/lewisbolding/Downloads/sasrh90.txt"), columns = colnames)
data_90_old = data_90_old.drop(columns='Year')
data_90_old.State = data_90_old.State.replace(state_idxs_for_file, state_names)

state_dist = np.zeros([51, 51]) #Generate state distance array
for st1idx in range(len(state_names)):
    st1_name = state_names[st1idx]
    for st2idx in range(len(state_names)):
        st2_name = state_names[st2idx]
        if st1idx == st2idx:
            state_dist[st1idx, st2idx] = 0
        else:
            list_cities_st1 = reduced_list_cities.loc[reduced_list_cities.state_name == st1_name]
            list_cities_st2 = reduced_list_cities.loc[reduced_list_cities.state_name == st2_name]
            dists = []
            for city1idx in list_cities_st1.index:
                for city2idx in list_cities_st2.index:
                    lat1 = np.deg2rad(list_cities_st1.loc[city1idx].lat)
                    lng1 = np.deg2rad(list_cities_st1.loc[city1idx].lng)
                    lat2 = np.deg2rad(list_cities_st2.loc[city2idx].lat)
                    lng2 = np.deg2rad(list_cities_st2.loc[city2idx].lng)
                    dist = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lng2-lng1)) * 6371
                    dists = np.append(dists, dist)
            mindist = np.min(dists) #Find shortest city-city distance between states
            state_dist[st1idx, st2idx] = mindist

age_groups = np.array(["<16", "16-24", "25-44", "45-64", ">64"])
agent_sexes = ["Male", "Female"]

data_90_old_2 = pd.DataFrame(columns = colnames[1:])
for state in state_names:
    state_idx = data_90_old.State == state
    for age_grp in age_groups:
        if age_grp == "<16":
            age_idx = data_90_old.Age < 16
        elif age_grp == "16-24":
            age_idx = (data_90_old.Age >= 16) & (data_90_old.Age < 25)
        elif age_grp == "25-44":
            age_idx = (data_90_old.Age >= 25) & (data_90_old.Age < 45)
        elif age_grp == "45-64":
            age_idx = (data_90_old.Age >= 45) & (data_90_old.Age < 64)
        elif age_grp == ">64":
            age_idx = data_90_old.Age > 64
        r_data = data_90_old[state_idx & age_idx]
        r_data = pd.DataFrame(r_data.sum()).T
        r_data.State = state
        r_data.Age = age_grp
        data_90_old_2 = pd.concat([data_90_old_2, r_data])

data_90 = pd.DataFrame(columns = ["State", "Age", "Male", "Female"])
data_90.State = data_90_old_2.State
data_90.Age = data_90_old_2.Age
data_90.Male = np.sum(data_90_old_2.iloc[:, [2, 4, 6, 8, 10, 12, 14, 16]], axis = 1)
data_90.Female = np.sum(data_90_old_2.iloc[:, [3, 5, 7, 9, 11, 13, 15, 17]], axis = 1)


df_2023 = pd.read_excel("/Users/lewisbolding/project/data/ACSDP1Y2023.DP05-2025-02-17T150049.xlsx")
total_pop_2023_with_age = df_2023.iloc[5:18]
idx_range_for_2023 = 1 + np.arange(0, 51) * 4
values_2023_with_age = total_pop_2023_with_age.iloc[:, idx_range_for_2023]
values_2023_with_age = values_2023_with_age.replace(',', '', regex=True).astype(float)
summed_values_2023 = np.zeros([51, 5])
summed_values_2023[:, 0] = np.sum(values_2023_with_age.iloc[0:3])
summed_values_2023[:, 1] = np.sum(values_2023_with_age.iloc[3:5])
summed_values_2023[:, 2] = np.sum(values_2023_with_age.iloc[5:7])
summed_values_2023[:, 3] = np.sum(values_2023_with_age.iloc[7:10])
summed_values_2023[:, 4] = np.sum(values_2023_with_age.iloc[10:13])
data_age_23 = summed_values_2023

df_2023_2 = pd.read_excel("/Users/lewisbolding/Downloads/SCPRC-EST2023-18+POP.xlsx")
data_23 = df_2023_2.iloc[8:59, 1]

def total_pop_init(state): #Find total population of each subgroup in 1990
    stateidxs = data_90.State == state
    usefuldata = data_90[stateidxs]
    usefuldata = usefuldata.drop(columns=['Age', 'State'])
    tot = usefuldata.sum().sum()
    return tot, usefuldata

def num_agents_init(state, agent_sex, age_group):
    tot = total_pop_init(state)[0]
    data = total_pop_init(state)[1]
    num_type_all_ages = np.array(data[agent_sex])
    age_idx = age_group == age_groups
    num_type = num_type_all_ages[age_idx][0]
    return num_type / fidelity

def eval_population(keyword): #Generate dataframe holding all agents' information. This has three functions
    global pop_df
    global ags
    if keyword == "init": #Initialise
        population_df = pd.DataFrame(columns = ["State", "Age", "Sex"])
    elif keyword == "refresh": #Update agents' location
        population_df = pop_df
        reduced_ags = ags[:len(population_df)]
        states = [i.state for i in reduced_ags]
        population_df["State"][range(len(reduced_ags))] = states
    elif keyword == "addnew": #Add newly generated agents to dataframe
        population_df = pop_df
        reduced_ags = ags[len(population_df):]
        states = [i.state for i in reduced_ags]
        sexes = [i.sex for i in reduced_ags]
        ages = [i.age for i in reduced_ags]
        demos = (np.vstack([states, ages, sexes])).T
        demos_df = pd.DataFrame(demos, columns = ["State", "Age", "Sex"])
        population_df = pd.concat([population_df, demos_df])
        population_df = population_df.reset_index().drop(columns="index")
    return population_df

def total_pop(state): #Find total population of each subgroup
    global pop_df
    totalp = int(np.round(np.sum(pop_df.State == state))) * fidelity
    return totalp

def num_agents(state, age, agent_sex):
    global pop_df
    agent_idx = (pop_df.State == state) & (pop_df.Age == age) & (pop_df.Sex == agent_sex)
    num = int(np.ceil(np.sum(agent_idx))) * fidelity
    return num

#%% BIRTHS DATA

m1990 = pd.read_csv("/Users/lewisbolding/Downloads/data90.txt", delim_whitespace=True, header=None).reset_index() #10-19
b1990 = pd.read_csv("/Users/lewisbolding/Downloads/births90.txt", delim_whitespace=True, header=None).reset_index() #10-19
d1990 = pd.read_csv("/Users/lewisbolding/Downloads/deaths_1990.txt", delim_whitespace=True, header=None).reset_index() #10-19
p1990 = pd.read_csv("/Users/lewisbolding/Downloads/pop90.txt", delim_whitespace=True, header=None).reset_index() #10-19
migr_1990 = m1990.iloc[:, 5:]
migr_1990 = migr_1990.iloc[:, ::-1]
migr_1990.iloc[:, 0] = migr_1990.iloc[:, 0] * 4
births_1990 = b1990.iloc[:, 5:]
births_1990 = births_1990.iloc[:, ::-1]
births_1990.iloc[:, 0] = births_1990.iloc[:, 0] * 4
deaths_1990 = d1990.iloc[:, 5:]
deaths_1990 = deaths_1990.iloc[:, ::-1]
deaths_1990.iloc[:, 0] = deaths_1990.iloc[:, 0] * 4
pop_1990 = p1990.iloc[:, 4:-1]
pop_1990 = pop_1990.iloc[:, ::-1]

xcx2000 = pd.read_csv("/Users/lewisbolding/Downloads/data00.txt") #10-19
xcx_idxs = xcx2000.STNAME == xcx2000.CTYNAME
xcx2000_reduced = xcx2000[xcx_idxs].drop(328)
births_2000 = (xcx2000_reduced.reset_index()).iloc[:, 32:43]
births_2000.iloc[:, 0] = births_2000.iloc[:, 0] * 4
deaths_2000 = (xcx2000_reduced.reset_index()).iloc[:, 43:54]
deaths_2000.iloc[:, 0] = deaths_2000.iloc[:, 0] * 4
migr_2000 = (xcx2000_reduced.reset_index()).iloc[:, 65:76]
migr_2000.iloc[:, 0] = migr_2000.iloc[:, 0] * 4
pop_2000 = (xcx2000_reduced.reset_index()).iloc[:, 10:21]

xcx2010 = pd.read_csv("/Users/lewisbolding/Downloads/data1.txt") #10-19
births_2010 = xcx2010.iloc[5:56, 28:37].reset_index().iloc[:, 1:]
deaths_2010 = xcx2010.iloc[5:56, 38:47].reset_index().iloc[:, 1:]
migr_2010 = xcx2010.iloc[5:56, 58:67].reset_index().iloc[:, 1:]
pop_2010 = xcx2010.iloc[5:56, 7:16].reset_index().iloc[:, 1:]

xcx2020 = pd.read_csv("/Users/lewisbolding/Downloads/data24.txt")
births_2020 = xcx2020.iloc[14:65, 16:21].reset_index().iloc[:, 1:]
births_2020.iloc[:, 0] = births_2020.iloc[:, 0] * 4
deaths_2020 = xcx2020.iloc[14:65, 21:26].reset_index().iloc[:, 1:]
deaths_2020.iloc[:, 0] = deaths_2020.iloc[:, 0] * 4
migr_2020 = xcx2020.iloc[14:65, 31:36].reset_index().iloc[:, 1:]
migr_2020.iloc[:, 0] = migr_2020.iloc[:, 0] * 4
pop_2020 = xcx2020.iloc[14:65, 6:11].reset_index().iloc[:, 1:]

births_data = pd.concat([births_1990, births_2000, births_2010, births_2020], ignore_index=True, axis=1)
deaths_data = pd.concat([deaths_1990, deaths_2000, deaths_2010, deaths_2020], ignore_index=True, axis=1)
migr_data = pd.concat([migr_1990 * 1.348, migr_2000 * 1.365, migr_2010 * 1.268, migr_2020], ignore_index=True, axis=1) 
pop_data = pd.concat([pop_1990, pop_2000, pop_2010, pop_2020], ignore_index=True, axis=1)


# %% CLIMATE DATA AND FUNCTIONS

rates = pd.read_excel('/Users/lewisbolding/Downloads/perc.xlsx')
rates_2 = np.array(rates.Rate)

rates_by_year = np.array([1.13, 1.34, 1.39, 1.32, 1.23, 1.19, 1.16, 1.20, 1.17, 1.15, 1.11, 0.98, 0.93, 0.86,
                          0.93, 0.92, 0.96, 0.95, 0.95, 0.88, 0.83, 0.73, 0.73, 0.69, 0.73, 0.74, 0.72, 0.63,
                          0.53, 0.46, 0.97, 0.16, 0.37, 0.49])

rates_by_year = np.append(rates_by_year, rates_2[34:]) #Population growth rates array

clim_perc = np.array([96, 6, 62, 88, 57, 20, 22, 66, 34, 70, #Climate vulnerability index values (see report))
                      92, 0, 30, 54, 76, 32, 60, 90, 98, 40,
                      38, 50, 52, 14, 100, 64, 24, 28, 42, 4,
                      56, 82, 48, 78, 8, 68, 74, 18, 72, 44,
                      94, 36, 84, 86, 12, 2, 26, 10, 80, 16,
                      46])

health_clim =    np.array([98, 0, 64, 72, 60, 38, 36,
                  76, 68, 70, 90, 26, 14,
                  52, 44, 30, 40, 100, 84,
                  4, 82, 54, 24, 42, 80,
                  48, 6, 28, 46, 18, 74,
                  62, 57, 94, 10, 66, 56,
                  20, 78, 34, 96, 22, 86,
                  50, 8, 2, 88, 32, 92,
                  16, 12])

social_clim =    np.array([86, 8, 12, 84, 32, 34, 28,
                  18, 2, 76, 62, 0, 60,
                  46, 70, 100, 78, 82, 94,
                  50, 26, 36, 14, 54, 92,
                  52, 44, 96, 4, 22, 40,
                  66, 24, 64, 74, 56, 38,
                  20, 88, 16, 68, 98, 48,
                  90, 57, 10, 30, 6, 72,
                  42, 80])

extreme_clim =  np.array([70, 8, 22, 44, 16, 57, 94,
                 82, 10, 40, 52, 0, 80,
                 20, 24, 56, 32, 24, 46,
                 100, 74, 96, 38, 42, 64,
                 14, 92, 18, 2, 98, 86,
                 84, 12, 62, 60, 28, 36,
                 4, 68, 76, 54, 72, 26,
                 66, 78, 88, 50, 6, 48,
                 30, 90])

health_comm =   np.array([94, 44, 62, 96, 8, 32, 2,
                 70, 48, 72, 82, 6, 38,
                 46, 88, 30, 57, 86, 100,
                 56, 20, 18, 64, 0, 98,
                 76, 42, 24, 66, 16, 26,
                 60, 22, 74, 14, 80, 84,
                 50, 52, 54, 90, 10, 92,
                 68, 4, 12, 34, 28, 78,
                 36, 40])

infra_comm =    np.array([92, 68, 38, 90, 34, 4, 36,
                 24, 57, 64, 94, 10, 50,
                 62, 78, 32, 72, 86, 98,
                 48, 14, 18, 56, 8, 100,
                 80, 52, 44, 46, 22, 6,
                 76, 40, 70, 16, 66, 82,
                 2, 42, 30, 88, 54, 96,
                 84, 0, 28, 26, 12, 74,
                 20, 60])

social_comm =   np.array([72, 90, 96, 86, 92, 8, 12,
                 50, 74, 70, 62, 76, 40,
                 34, 57, 14, 46, 82, 80,
                 38, 22, 36, 42, 18, 94,
                 52, 24, 28, 98, 0, 32,
                 100, 68, 54, 4, 56, 80,
                 84, 30, 44, 78, 20, 60,
                 64, 2, 10, 6, 48, 66,
                 16, 26])

env_comm =      np.array([32, 4, 84, 14, 100, 68, 52,
                 82, 70, 46, 62, 8, 12,
                 96, 80, 24, 72, 28, 78,
                 2, 48, 54, 94, 50, 38,
                 60, 10, 36, 64, 6, 92,
                 18, 90, 42, 20, 66, 40,
                 56, 98, 74, 44, 16, 26,
                 86, 88, 0, 22, 57, 34,
                 76, 30])

clim_perc_fix = social_comm/200 #Adjust chosen climate factor (see report)


#%% BIRTHS/DEATHS/MIGR SPLIT

year_start = 1990 #Choose start and end date for base growth values
year_end = 2010

year_start_idx = year_start - 1990
year_end_idx = year_end - 1990

length_years = year_end - year_start

births_split = births_data.iloc[:, year_start_idx:year_start_idx+length_years].sum(axis=1)
deaths_split = deaths_data.iloc[:, year_start_idx:year_start_idx+length_years].sum(axis=1)
nat_growth_split = births_split - deaths_split
migration_split = migr_data.iloc[:, year_start_idx:year_start_idx+length_years].sum(axis=1)
total_split = pop_data.iloc[:, year_start_idx]

gro_nat = np.zeros(51) #Find average natural growth rate for each state over chosen period
for stidx in range(len(state_names)):
    tot_0 = total_split[stidx]
    tot_1 = (tot_0 + nat_growth_split[stidx])
    gr_r = (tot_1 / tot_0) ** (1/length_years)
    gro_nat[stidx] = gr_r
    
gro_migr = np.zeros(51) #Find average international migration growth rate for each state over chosen period
for stidx in range(len(state_names)):
    tot_0 = total_split[stidx]
    tot_1 = (tot_0 + migration_split[stidx])
    gr_r = (tot_1 / tot_0) ** (1/length_years)
    gro_migr[stidx] = gr_r

rates_file = pd.read_excel("/Users/lewisbolding/Downloads/np2023-t1.xlsx")
rates_file = rates_file.iloc[7:85]
rates_hist_nat = rates_by_year[:35] * (births_data.sum() - deaths_data.sum()) / (births_data.sum() - deaths_data.sum() + migr_data.sum())
rates_hist_migr = rates_by_year[:35] * migr_data.sum()  / (births_data.sum() - deaths_data.sum() + migr_data.sum())
rates_proj_nat = (rates_file.iloc[2:, 6]/rates_file.iloc[2:, 1]) * 100
rates_proj_migr = (rates_file.iloc[2:, 7]/rates_file.iloc[2:, 1]) * 100
rates_by_year_nat = np.concatenate([rates_hist_nat, rates_proj_nat])
rates_by_year_migr = np.concatenate([rates_hist_migr, rates_proj_migr]) #Find total historical/projected growth rates

def find_growth_multiplier_nat(year): #Generate scaling factor for state-specific growth values (see report)
    rate_1 = rates_by_year_nat[year]
    rate_2 = ((np.sum(total_split + nat_growth_split)/np.sum(total_split)) ** (1/length_years) - 1) * 100
    mult = (rate_1 / rate_2)
    return mult

def find_growth_multiplier_migr(year):
    rate_1 = rates_by_year_migr[year]
    rate_2 = ((np.sum(total_split + migration_split)/np.sum(total_split)) ** (1/length_years) - 1) * 100
    mult = (rate_1 / rate_2)
    return mult

# %% GRAVITY MODEL

G = 2.89e-6 #Coefficients, determined using least squares multiple regression
P1_coeff = 0.809
P2_coeff = 0.758
dist_coeff = -0.601

def move_num(initial_state, final_state, year):  #Find expected number of movers M_ij for a given year
    initial_state_idx = np.where(state_names == initial_state)[0][0]
    P1 = total_pop(initial_state)
    final_state_idx = np.where(state_names == final_state)[0][0]
    P2 = total_pop(final_state)
    dist = state_dist[initial_state_idx, final_state_idx]
    num_moved = G * (P1 ** P1_coeff) * (P2 ** P2_coeff) * (dist ** dist_coeff) * (1 - 0.13 + clim_perc_fix[initial_state_idx] - clim_perc_fix[final_state_idx]) ** ((projection[year] - projection[0]))                                                                                                                                                                     
    if initial_state_idx == 4:
        if year < 33:
         num_moved = num_moved * 2.03456 #Specific adjustments for transient economic factors
    sunbelt = ["Alabama", "Arizona", "Arkansas", "California", "Colorado", "Florida", "Georgia", "Louisiana", "Mississippi", "New Mexico", "Nevada", "North Carolina", "Oklahoma", "South Carolina", "Tennessee", "Texas"]
    sunbelt_idxs = [np.where(state_names==state)[0] for state in sunbelt]
    if final_state_idx in sunbelt_idxs:
        if initial_state_idx not in sunbelt_idxs:
            if year < 30:
                num_moved = num_moved * 1.6
    return num_moved

def generate_prob_array(year): #Convert M_ij values into array of probabilities
    moved_array = np.zeros([51, 51])
    for initial_state_idx_loop in range(len(state_names)):
        initial_state_name = state_names[initial_state_idx_loop]
        for final_state_idx_loop in range(len(state_names)):
            final_state_name = state_names[final_state_idx_loop]
            if initial_state_idx_loop == final_state_idx_loop:
                moved_array[initial_state_idx_loop, final_state_idx_loop] = 0
            else:
                moved_array[initial_state_idx_loop, final_state_idx_loop] = move_num(initial_state_name, 
                                                                                  final_state_name, year)

        total_moved = np.sum(moved_array[initial_state_idx_loop, :])
        total_stayed = total_pop(initial_state_name) - total_moved
        moved_array[initial_state_idx_loop, initial_state_idx_loop] = total_stayed
    moved_array_cum = np.cumsum(moved_array, axis = 1)

    for initial_state_idx_loop in range(len(state_names)):
        moved_array_cum[initial_state_idx_loop, :] = moved_array_cum[initial_state_idx_loop, :] / moved_array_cum[initial_state_idx_loop, -1]
    return moved_array_cum

 # %% AGENT SETUP

class agent: #Define agent class
    def __init__(self, origin_state, state_history, agent_sex, agent_age, group_id):
        self.state = origin_state #Initial state before any migration decisions are made
        self.history = state_history #Record of where agents move year-by-year
        self.sex = agent_sex
        self.age = agent_age
        self.id = group_id

pd.options.mode.chained_assignment = None
add = 0

lost_agents = data_90.iloc[:, :].copy(deep=True)
lost_agents.iloc[:, 2:] = 0

reduced_age_groups = age_groups[1:]

ags = np.array([]) #Create empty list of agents
for idx_state in list_states.index: #Loop over 51 states
    state = state_names[idx_state] #Find state name
    state_history = state #Generate state history array (appended later)
    for age_grp in age_groups:
        lost_agents_row_idx = (lost_agents.State == state) & (lost_agents.Age == age_grp)
        for sex_ag in agent_sexes:
            num_1_b = num_agents_init(state, sex_ag, age_grp)
            num_1 = round(num_1_b)
            lost_agents.loc[lost_agents_row_idx, sex_ag] = num_1_b - num_1 #Find number of agents "rounded out", store for later
            add = add + num_1
            if num_1 != 0:
                new_agent = agent(state, state_history, sex_ag, age_grp, 0)
                new_agents = [copy.deepcopy(new_agent) for _ in range(num_1)]
                ags = np.append(ags, new_agents) #Add new agent to list


pop_df = eval_population("init")
pop_df = eval_population("addnew")

pop_df_1990 = pop_df.copy(deep=True)
ags_1990 = copy.deepcopy(ags)

# %% RUN MODEL

pop_dfs = list() #Store yearly dataframes in array

ags = copy.deepcopy(ags_1990) #Start from initial population of agents
pop_df = pop_df_1990.copy(deep=True)

num_in_year = np.array([len(ags)]) #Track total population over time
num_in_year_state = np.zeros(51) #Track total population in each state over time
for i in range(len(state_names)):
    state = state_names[i]
    num_in_year_state[i] = np.sum(pop_df.State == state)
    
for year in range(num_years): #Simulate over N years
    year_actual = str(1990 + year)

    prob_array = generate_prob_array(year) #Generate probability array
    for state_idx in range(51):
        state = state_names[state_idx]
        ags_in_state = ags[pop_df.State == state]
        rand_cs = np.random.rand(len(ags_in_state)) #Assign all agents in state random value between 0 and 1
        state_destination_idxs = np.searchsorted(prob_array[state_idx, :], rand_cs) 
        state_destination_names = np.array(state_names[state_destination_idxs]) #These random numbers correspond to a destination state
       
        for ag_idx in range(len(ags_in_state)): #Loop through agents
            i = ags_in_state[ag_idx]
            i.state = state_destination_names[ag_idx]
            i.history = np.append(i.history, state_destination_names[ag_idx]) #Move agents to new state
                
    pop_df = eval_population("refresh") #Update with new locations

    multiplier_nat = find_growth_multiplier_nat(year) #Find natural and migration growth figures
    multiplier_migr = find_growth_multiplier_migr(year)
    for idx_state in list_states.index:
        state = state_names[idx_state] 
        state_history = np.append(np.repeat("None",year + 1), state) #New agents have "None" tags for time steps before they are generated
        num_in_state = np.sum(pop_df.State == state)
        for age_grp in age_groups:
            lost_agents_row_idx = (lost_agents.State == state) & (lost_agents.Age == age_grp)
            for sex_ag in agent_sexes:
                num_2_b = num_agents(state, age_grp, sex_ag) * (((gro_migr[idx_state] - 1) * multiplier_migr + (gro_nat[idx_state] - 1) * multiplier_nat) / fidelity) + lost_agents[lost_agents_row_idx][sex_ag].loc[0]
                num_2 = int(round(num_2_b))
                if num_2 <= 0: #Agents cannot be deleted
                    num_2 = 0
                lost_agents.loc[lost_agents_row_idx, sex_ag] = num_2_b - num_2
                if num_2 != 0:
                    new_agent = agent(state, state_history, sex_ag, age_grp, 0)
                    new_agents = [copy.deepcopy(new_agent) for _ in range(num_2)]
                    ags = np.append(ags, new_agents) #Add new agent to list

    pop_df = eval_population("addnew")                

    num_in_year = np.append(num_in_year, len(ags))
    num_in_year_state_temp = np.zeros(51)
    for i in range(len(state_names)):
        state = state_names[i]
        num_in_year_state_temp[i] = np.sum(pop_df.State == state)
    num_in_year_state = np.vstack([num_in_year_state, num_in_year_state_temp])
                                
    pop_dfs.append(pop_df)