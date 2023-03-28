
# imports
from ma_abm.agent import Person
from ma_abm.model import VirusModel, VirusModel_mvp

# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython


parameters = {
    'my_parameter':42,
    'agents':1000,
    'steps':50,
    'wealth_appeal_corr': 0, # >0 more wealth higher appeal chance
    'acc': 0.8, # accuracy of fraud prdediction
    'conviction_rate': 1,
    'appeal_wealth': 0.2 # minimal wealth needed for appeal (could also become a param for distr. eventually)
    #'wealth_impact'
    
}

model = VirusModel(parameters)
results = model.run() 

df = results['variables']['Person']
fig, ax = plt.subplots()

sns.lineplot(ax = ax,
             data = df,
             x = df.index.get_level_values('t'),
             y = df['wealth'].astype(float),
             hue = df['race'],
             marker = 'o')

plt.show()


model = VirusModel_mvp(parameters)
results_mvp = model.run() 

df = results['variables']['Person']
fig, ax = plt.subplots()

sns.lineplot(ax = ax,
             data = df,
             x = df.index.get_level_values('t'),
             y = df['wealth'].astype(float),
             hue = df['race'],
             marker = 'o')

plt.show()


