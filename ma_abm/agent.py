# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython


class Person(ap.Agent):
    
    def setup(self):  
        """ Initialize a new variable at agent creation. """
        # self.condition = 0  # Susceptible = 0, Infected = 1, Recovered = 2
        a = 5 # shape
        rng = np.random.default_rng()
        
        
        # race
        self.race =  rng.binomial(1,0.2)#binary not white0.2 /  white for the moment 0.8
        
        if self.race == 0:
            self.wealth = rng.beta(1.5, 5,1)[0]
        else:
            self.wealth = rng.beta(5, 3,1)[0]
            

        # fraud
        self.fraud = rng.binomial(1,0.5,1)[0]
        self.fraud_pred = -1
        self.convicted = 0

        
        
        
    def fraud_algo(self):
        """ DM mechanism can also be ML"""
        rng = np.random.default_rng()
        # self.fraud_pred = rng.binomial(1, 0.5)
        if self.fraud == 1:
            fraud_cor = rng.binomial(1,self.p.acc)
        else:
            fraud_cor = rng.binomial(1,1-self.p.acc)
            
        self.fraud_pred = rng.binomial(1, fraud_cor*(0.8-self.p.wealth_appeal_corr))
        

            
    def convict(self):
        """ Conviction and Consequences"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1:
#             if rng.binomial(1,0.8) == 1:
                # pay fine, get on record, 
            self.wealth = self.wealth - np.max([0.01,(pow(self.wealth,2)*0.001)])
            self.convicted =+ 1
            self.fraud = rng.binomial(1,0.5,1)[0]
            self.fraud_pred = 0
            
            
    def convict_true(self):
        """ Conviction and Consequences"""
        rng = np.random.default_rng()
        if self.fraud == 1:
#             if rng.binomial(1,0.8) == 1:
            # pay fine, get on record, 
            self.wealth = self.wealth - (0.01)#(self.wealth*0.05)])
            self.convicted =+ 1
            self.fraud = rng.binomial(1,0.5,1)[0]
            self.fraud_pred = 0

    def wealth_grow(self):
        self.wealth = min(1,self.wealth+pow(self.wealth,2)*0.01)
            
