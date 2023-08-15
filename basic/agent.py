# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np
import sklearn
import pickle
import pandas as pd
import os

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython

#import other functions
from utils import generate_init

from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio


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
        self.race,self.gender,self.wealth,self.health,self.fraud,self.fraud_pred,self.convicted = generate_init(train_clf = False, n =1)
        self.dpd_race = 0
        self.dpd_gender = 0
        self.eod_race = 0
        self.eod_gender = 0
        
        
    def fraud_algo(self, classifier = True):
        """ DM mechanism can also be ML"""

        path = os.path.abspath(os.getcwd())
        self.resources = self.wealth
        self.fraud_pred =0

        # decide how much influence the resources have 
        res_weight = 0.3

        if classifier != 'None':
            agent = [[self.race, self.gender, self.wealth, self.health]]
            # filename = ("/gpfs/home4/ltiyavorabu/abm/basic/"+classifier)
            filename = classifier
            agent = pd.DataFrame(agent, columns = ['race', 'gender', 'wealth', 'health'])
            with open(filename, "rb") as f:
                clf = pickle.load(f)    
            self.fraud_pred = clf.predict(agent)[0]
            # self.fraud_pred = np.rint(self.fraud_pred[0]) 

        else:
            rng = np.random.default_rng()
            if self.fraud == 1:
                self.fraud_pred = rng.binomial(1, self.p.acc)
            else:
                self.fraud_pred = rng.binomial(1, (1-self.p.acc))

        return self.fraud_pred
        

#  if classifier != None:

#             agent = [[self.race, self.gender, self.wealth, self.health]]
#             with open("clf.pkl", "rb") as f:
#                 clf = pickle.load(f)    
#             self.fraud_pred = ((1- res_weight)*clf.predict_proba(agent) + res_weight* self.resources)[0]
#             self.fraud_pred = np.rint(self.fraud_pred[0])

#         else:
#             rng = np.random.default_rng()
#             if self.fraud == 1:
#                 self.fraud_pred = rng.binomial(1, ((1- res_weight)*self.p.acc + res_weight* self.resources))
#             else:
#                 self.fraud_pred = rng.binomial(1, (1- res_weight)*(1-self.p.acc) + res_weight* self.resources)
        



        # print(self.fraud_pred)

        ### for more elaborate modelling ###
        # self.fraud_pred = rng.binomial(1, fraud_cor) #*(0.8-self.p.wealth_appeal_corr))
        

    def appeal(self):
        """Possibility to Appeal to Fraud Algo Decision"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1 and self.wealth > self.p.appeal_wealth:
            self.fraud_algo(self.p.clf)
            
    def convict(self):
        """ Conviction and Consequences"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1:
            self.wealth = np.clip(self.wealth - np.max([0.05,(self.wealth*0.1)]),0,1)
            self.convicted =+ 1
        self.fraud = rng.binomial(1,np.clip(((self.wealth-0.75)**4+0.3), 0,0.9))
            # self.fraud_pred = 0
    
    def wealth_grow(self):
        self.wealth = min(1,self.wealth+pow(self.wealth,2)*0.1)


    # def bureaucratics(self):
    #     """ This function is a collection of smaller functions that determine the threshold that the bureaucratic process imposes on people.
    #         Takes resources and applies function that weighs impact of resources."""

    #     if self.pers_resources 


    def fairness_metrics(self,data, print = False):

        y_true = list(data.fraud)
        y_pred = list(data.fraud_pred)
        gender = data.gender
        race = data.race

        # print(y_true)
        # print(sum(y_pred))


        dpd = []
        eod = []
        

        for i in [gender,race]:
            if (sum(y_true) != 0 and sum(y_pred) != 0):
                try:
                    temp_dpd = demographic_parity_ratio( y_true=y_true, y_pred=y_pred, sensitive_features=i)
                except ZeroDivisionError:
                    temp_dpd = 0
                dpd.append(temp_dpd)
                try:
                    temp_eod = equalized_odds_ratio( y_true=y_true, y_pred=y_pred, sensitive_features=i)
                except ZeroDivisionError:
                    temp_eod = 0
                if print:
                    print('dpd',temp_dpd)
                    print('eod',temp_eod)
                eod.append(temp_eod)


        # dpd = demographic_parity_difference( y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        if (sum(y_true) != 0 and sum(y_pred) != 0):
            if not print:
                self.eod_gender = eod[0]
                self.eod_race = eod[1]
                self.dpd_gender = dpd[0]
                self.dpd_race = dpd[1]   
            
        


#     def step(self):
#         # The agent's step will go here.
#         # For demonstration purposes we will print the agent's unique_id
#         self.appeal()
#         print("Hi, I am agent " + str(self.unique_id) + ".")
#         # print("my wealth, job, fraud, fraud_pred is:" + str(self.wealth)+ str(self.job) + str(self.fraud)+ str(self.fraud_pred))