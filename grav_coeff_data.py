#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:42:19 2025

@author: lewisbolding
"""

import pandas as pd
import numpy as np

colnames1 = ["State", "US", "Alabama", "Alaska",  "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware"]
df1 = pd.read_csv("/Users/lewisbolding/project/migr1.txt", sep='\s+', names = colnames1, converters={"State": str})
df1 = df1.drop(columns="US")

colnames2 = ["State", "District of Columbia", "Florida", "Georgia",  "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas"]
df2 = pd.read_csv("/Users/lewisbolding/project/migr2.txt", sep='\s+', names = colnames2, converters={"State": str})

colnames3 = ["State", "Kentucky", "Louisiana", "Maine",  "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri"]
df3 = pd.read_csv("/Users/lewisbolding/project/migr3.txt", sep='\s+', names = colnames3, converters={"State": str})

colnames4 = ["State", "Montana", "Nebraska", "Nevada",  "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota"]
df4 = pd.read_csv("/Users/lewisbolding/project/migr4.txt", sep='\s+', names = colnames4, converters={"State": str})

colnames5 = ["State", "Ohio", "Oklahoma", "Oregon",  "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas"]
df5 = pd.read_csv("/Users/lewisbolding/project/migr5.txt", sep='\s+', names = colnames5, converters={"State": str})

colnames6 = ["State", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
df6 = pd.read_csv("/Users/lewisbolding/project/migr6.txt", sep='\s+', names = colnames6, converters={"State": str})

df7 = df1.set_index('State').join(df2.set_index('State'))
df8 = df7.join(df3.set_index('State'))
df9 = df8.join(df4.set_index('State'))
df10 = df9.join(df5.set_index('State'))
df_fin = df10.join(df6.set_index('State'))
df_fin = df_fin.reset_index()

df_fin.State = list_states_grav