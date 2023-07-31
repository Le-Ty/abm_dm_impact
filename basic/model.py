
# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np
import pickle

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython

from agent import Person
from utils import classifier_train



class VirusModel_baseline(ap.Model):
    
    def setup(self, n_train = 100): #before
        """ Initialize the agents and network of the model. """
        
        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Person)


        # first classifier trained on same distribution
        #load distributions
        if self.p.clf == 'hist':
            with open("data/distributions_init.pickle", "rb") as f:
                d_fnw = pickle.load(f)
                d_mw = pickle.load(f)
                d_mnw = pickle.load(f)
                d_fw = pickle.load(f)
            with open("data/values_init.pickle", "rb") as f:
                v_fnw = pickle.load(f)
                v_mw = pickle.load(f)
                v_mnw = pickle.load(f)
                v_fw = pickle.load(f)

                rng = np.random.default_rng()        
                
                # race
                x = []
                race =  rng.binomial(1,0.2,n_train) #binary not white0.2 /  white for the moment 0.8
                for i in race:
                    gender = rng.binomial(1,0.5,1)[0]
                    health = np.random.uniform(0,1,1)[0]

                    if (gender == 0 and i == 1).all():
                        wealth = (random.choices(v_fnw, weights=d_fnw, k=1))[0]
                    elif (gender == 1 and i == 1).all():
                        wealth = (random.choices(v_mnw, weights=d_mnw, k=1))[0]
                    elif (gender == 1 and i == 0).all():
                        wealth = (random.choices(v_mw, weights=d_mw, k=1))[0]
                    elif (gender == 0 and i == 0).all():
                        wealth = (random.choices(v_fw, weights=d_fw, k=1))[0]
                    # print(gender,i)
                    x.append([i,gender,wealth,health])


                # fraud
                y = rng.binomial(1,0.5,n_train)
                classifier_train(x, y)

        

        # create distribution & development analysis
        # average wealth of agent by race a t = 0
        # self.w_wealth_t0 = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        # self.nw_wealth_t0 = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
    

    def step(self, clf = 'hist'): # during each step
        """ Define the models' events per simulation step. """

        #EXECUTING FUNCTIONS


        #ACCESS & TREATMENT
        self.agents.fraud_algo(self.p.clf)
#         self.agents.appeal()


        #BUREAUCRATICS
        self.agents.convict()


        #SOCIETY
        self.agents.wealth_grow()





        # ALGO TRAINING

        if self.p.clf == 'update': # self.online_learning:
            # DM ALGO
            #create train/test data
            x = []
            y = []
            for ag in self.agents:
                x.append([ag.race, ag.gender, ag.wealth, ag.health])
                y.append(ag.fraud)

            # with open('clf_x.txt', 'wb') as fp:
            #     pickle.dump(x, fp)

            # with open('clf_y.txt', 'wb') as fp:
            #     pickle.dump(y, fp)

            # train classifier
            classifier_train(x, y)


        


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
        
        # w_wealth_tn = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        # nw_wealth_tn = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
        
        # w_wr_ratio = w_wealth_tn/self.w_wealth_t0
        # nw_wr_ratio = nw_wealth_tn/self.nw_wealth_t0

        # w_wr_ratio = w_wealth_tn - self.w_wealth_t0
        # nw_wr_ratio = nw_wealth_tn - self.nw_wealth_t0
        
        # self.report('w_wr_ratio', w_wr_ratio)
        # self.report('nw_wr_ratio', nw_wr_ratio)
            
        
        

class VirusModel(ap.Model):
    
    def setup(self): #before
        """ Initialize the agents and network of the model. """
        
        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Person)
    
        self.w_wealth_t0 = sum((self.agents.select(self.agents.race == 1)).wealth) / len((self.agents.select(self.agents.race == 1)))
        self.nw_wealth_t0 = sum((self.agents.select(self.agents.race == 0)).wealth) / len((self.agents.select(self.agents.race == 0)))
    

    def step(self): # during each step
        """ Define the models' events per simulation step. """
        self.agents.fraud_algo(self.p.clf)
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

        w_wr_ratio = w_wealth_tn - self.w_wealth_t0
        nw_wr_ratio = nw_wealth_tn - self.nw_wealth_t0
            
        
        
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
            