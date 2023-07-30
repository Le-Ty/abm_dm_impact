# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np
import sklearn
import pickle

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

        rng = np.random.default_rng()
        
        
        # race
        self.race =  rng.binomial(1,0.2)#binary not white0.2 /  white for the moment 0.8
        

        # gender
        self.gender =  rng.binomial(1,0.5)


        # wealth

        #load distributions
        with open("/data/distributions_init.pickle", "rb") as f:
            d_fnw = pickle.load(f)
            d_mw = pickle.load(f)
            d_mnw = pickle.load(f)
            d_fw = pickle.load(f)
        with open("/data/values_init.pickle", "rb") as f:
            v_fnw = pickle.load(f)
            v_mw = pickle.load(f)
            v_mnw = pickle.load(f)
            v_fw = pickle.load(f)



        if self.gender == 0 and self.race == 0:
            self.wealth = (random.choices(v_fnw, weights=d_fnw, k=1))
        elif self.gender == 1 and self.race == 0:
            self.wealth = (random.choices(v_mnw, weights=d_mnw, k=1))
        elif self.gender == 1 and self.race == 1:
            self.wealth = (random.choices(v_mw, weights=d_mw, k=1))
        elif self.gender == 0 and self.race == 1:
            self.wealth = (random.choices(v_fw, weights=d_fw, k=1))


        # health

        self.health = np.random.uniform(0,1,1)

        


        

        # fraud
        self.fraud = rng.binomial(1,0.5,1)[0]
        self.fraud_pred = -1
        self.convicted = 0

        
    def fraud_algo(self, classifier = True):
        """ DM mechanism can also be ML"""

        self.resources = self.wealth

        # decide how much influence the resources have 
        res_weight = 0.3


        if classifier:

            agent = [[self.wealth, self.race]]
            with open("clf.pkl", "rb") as f:
                clf = pickle.load(f)      
            
            self.fraud_pred = ((1- res_weight)*clf.predict_proba(agent) + res_weight* self.resources)[0]
            self.fraud_pred = np.rint(self.fraud_pred[0])
            print(self.fraud_pred)
        else:
            rng = np.random.default_rng()
            if self.fraud == 1:
                self.fraud_pred = rng.binomial(1, ((1- res_weight)*self.p.acc + res_weight* self.resources))
            else:
                self.fraud_pred = rng.binomial(1, (1- res_weight)*(1-self.p.acc) + res_weight* self.resources)
        
        # print(self.fraud_pred)

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


    # def bureaucratics(self):
    #     """ This function is a collection of smaller functions that determine the threshold that the bureaucratic process imposes on people.
    #         Takes resources and applies function that weighs impact of resources."""

    #     if self.pers_resources 


            
            
        


#     def step(self):
#         # The agent's step will go here.
#         # For demonstration purposes we will print the agent's unique_id
#         self.appeal()
#         print("Hi, I am agent " + str(self.unique_id) + ".")
#         # print("my wealth, job, fraud, fraud_pred is:" + str(self.wealth)+ str(self.job) + str(self.fraud)+ str(self.fraud_pred))