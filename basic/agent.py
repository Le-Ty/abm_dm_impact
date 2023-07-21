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
        """ 
        Initializes attributes of Agents, according to US census. 
        
        Attributes of Agents.
            --------
            race : list of floats
                cosine similarity of top & bottom 10 words in corpus
            gender : list of strings
                words with cosine similarity of top & bottom 10 words in corpus
            health : list of floats
                cosine similarity of top & bottom 10 words in corpus
            wealth : list of strings
                words with cosine similarity of top & bottom 10 words in corpus
            fraud : list of floats
                cosine similarity of top & bottom 10 words in corpus
        """

        # probability functions
        a = 5 # shape of beta function
        rng = np.random.default_rng()
        
        
        # race
        self.race =  rng.binomial(1,0.2)#binary not white0.2 /  white for the moment 0.8
        
        #wealth
        if self.race == 0:
            self.wealth = rng.beta(1.5, 5,1)[0]
        else:
            self.wealth = rng.beta(5, 3,1)[0]

        #health

        #gender

        
            

        # fraud
        self.fraud = rng.binomial(1,0.5,1)[0]
        self.fraud_pred = -1
        self.convicted = 0

        
    def fraud_algo(self):
        """ DM mechanism can also be ML"""
        rng = np.random.default_rng()
        if self.fraud == 1:
            self.fraud_pred = rng.binomial(1,self.p.acc)
        else:
            self.fraud_pred = rng.binomial(1,1-self.p.acc)
        
        ### for more elaborate modelling ###
        # self.fraud_pred = rng.binomial(1, fraud_cor) #*(0.8-self.p.wealth_appeal_corr))
        

    def appeal(self):
        """Possibility to Appeal to Fraud Algo Decision"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1 and self.wealth > self.p.appeal_wealth:
            self.fraud_algo()
            
    def convict(self):
        """ Conviction and Consequences"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1:
            self.wealth = self.wealth - np.max([0.01,(self.wealth*0.05)])
            self.convicted =+ 1
            self.fraud = rng.binomial(1,0.5,1)[0]
            self.fraud_pred = 0
    
    def wealth_grow(self):
        self.wealth = min(1,self.wealth+pow(self.wealth,2)*0.01)
            
            
        


#     def step(self):
#         # The agent's step will go here.
#         # For demonstration purposes we will print the agent's unique_id
#         self.appeal()
#         print("Hi, I am agent " + str(self.unique_id) + ".")
#         # print("my wealth, job, fraud, fraud_pred is:" + str(self.wealth)+ str(self.job) + str(self.fraud)+ str(self.fraud_pred))