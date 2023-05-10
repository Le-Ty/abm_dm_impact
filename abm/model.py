
# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython


class VirusModel_baseline(ap.Model):
    
    def setup(self): #before
        """ Initialize the agents and network of the model. """
        
        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Person)
    
        self.w_wealth_t0 = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        self.nw_wealth_t0 = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
    

    def step(self): # during each step
        """ Define the models' events per simulation step. """
        self.agents.fraud_algo()
#         self.agents.appeal()
        self.agents.convict()
        self.agents.wealth_grow()
        


    def update(self):  # after each step
        """ Record variables after setup and each step. """
        self.agents.record('wealth')
        self.agents.record('fraud_pred')
        self.agents.record('fraud')
        self.agents.record('race')
        
        
    
    def end(self):     
        """ Record evaluation measures at the end of the simulation. """
#         self.report('wealth', self.agents.wealth)
#         self.report('race', self.agents.race)
#         self.report('my_measure', 1)
        
        # record race wealth ratio 
        
        w_wealth_tn = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        nw_wealth_tn = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
        
        w_wr_ratio = w_wealth_tn/self.w_wealth_t0
        nw_wr_ratio = nw_wealth_tn/self.nw_wealth_t0
        
        self.report('w_wr_ratio', w_wr_ratio)
        self.report('nw_wr_ratio', nw_wr_ratio)
            
        
        
#         # Record final evaluation measures
#         self.report('Total share infected', self.I + self.R) 
#         self.report('Peak share infected', max(self.log['I']))


class VirusModel(ap.Model):
    
    def setup(self): #before
        """ Initialize the agents and network of the model. """
        
        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Person)
    
        self.w_wealth_t0 = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        self.nw_wealth_t0 = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
    

    def step(self): # during each step
        """ Define the models' events per simulation step. """
        self.agents.fraud_algo()
        self.agents.appeal()
        self.agents.convict()
        self.agents.wealth_grow()
        


    def update(self):  # after each step
        """ Record variables after setup and each step. """
        self.agents.record('wealth')
        self.agents.record('fraud_pred')
        self.agents.record('fraud')
        self.agents.record('race')
        
        
    
    def end(self):     
        """ Record evaluation measures at the end of the simulation. """
#         self.report('wealth', self.agents.wealth)
#         self.report('race', self.agents.race)
#         self.report('my_measure', 1)
        
        # record race wealth ratio 
        
        w_wealth_tn = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        nw_wealth_tn = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
        
        w_wr_ratio = w_wealth_tn/self.w_wealth_t0
        nw_wr_ratio = nw_wealth_tn/self.nw_wealth_t0
        
        self.report('w_wr_ratio', w_wr_ratio)
        self.report('nw_wr_ratio', nw_wr_ratio)
            
        
        
#         # Record final evaluation measures
#         self.report('Total share infected', self.I + self.R) 
#         self.report('Peak share infected', max(self.log['I']))



###################################################




# from .agent import Person


# class VirusModel(ap.Model):
    
#     def setup(self): #before
#         """ Initialize the agents and network of the model. """

# #         # Create agents and network
#         self.agents = ap.AgentList(self, self.p.agents, Person)
    
#         self.w_wealth_t0 = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
#         self.nw_wealth_t0 = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
    

#     def step(self): # during each step
#         """ Define the models' events per simulation step. """
#         self.agents.fraud_algo()
# #         self.agents.appeal()
#         self.agents.convict()
#         self.agents.wealth_grow()
        


#     def update(self):  # after each step
#         """ Record variables after setup and each step. """
#         self.agents.record('wealth')
#         self.agents.record('fraud_pred')
#         self.agents.record('fraud')
#         self.agents.record('race')
        
        
    
#     def end(self):     
#         """ Record evaluation measures at the end of the simulation. """
# #         self.report('wealth', self.agents.wealth)
# #         self.report('race', self.agents.race)
# #         self.report('my_measure', 1)
        
#         # record race wealth ratio 
        
#         w_wealth_tn = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
#         nw_wealth_tn = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
        
#         w_wr_ratio = w_wealth_tn/self.w_wealth_t0
#         nw_wr_ratio = nw_wealth_tn/self.nw_wealth_t0
        
#         self.report('w_wr_ratio', w_wr_ratio)
#         self.report('nw_wr_ratio', nw_wr_ratio)
            