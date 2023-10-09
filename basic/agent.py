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
from generate_data import generate_init
from generate_data import fraud_val, generate_bias

from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio
from fairlearn.reductions import DemographicParity


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
        self.race,self.gender,self.wealth,self.health,self.star,self.fraud,self.fraud_pred,self.convicted = generate_init(self.p.star_version, self.p.synth_data_acc, self.p.abm_eval, train_clf = False, n =1)
        self.dpd_race = 0
        self.dpd_gender = 0
        self.eod_race = 0
        self.eod_gender = 0
        self.eval_acc  =0
        
        
    def fraud_algo(self, classifier):
        """ DM mechanism can also be ML"""

        path = os.path.abspath(os.getcwd())


        # decide how much influence the resources have 
        res_weight = 0.3

        if classifier != None and classifier != 'None':
           
            filename = ("/gpfs/home4/ltiyavorabu/abm/basic/"+classifier)
            # filename = ("clfs/" + classifier)
            if self.p.star_version != None:
                agent = [[self.race, self.gender, self.wealth, self.health, self.star]]
                agent = pd.DataFrame(agent, columns = ['race', 'gender', 'wealth', 'health', 'star'])
            else:
                agent = [[self.race, self.gender, self.wealth, self.health]]
                agent = pd.DataFrame(agent, columns = ['race', 'gender', 'wealth', 'health'])
            
            with open(filename, "rb") as f:
                clf = pickle.load(f)    

            # temp = self.fraud_pred
            self.fraud_pred = clf.predict(agent)[0]
            # print(temp -self.fraud_pred)


        else:
            rng = np.random.default_rng()
            if self.fraud == 1:
                self.fraud_pred = rng.binomial(1, self.p.acc)
            else:
                self.fraud_pred = rng.binomial(1, (1-self.p.acc))


    def update_star(self):
        temp = self.star
        self.star = generate_bias(self.race, self.gender, self.wealth, self.health, self.fraud, self.p.star_version, self.p.synth_data_acc)[0]

        

    def appeal(self):
        """Possibility to Appeal to Fraud Algo Decision"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1:
            if self.p.appeal_inter:
                if self.wealth > (self.p.appeal_wealth) and self.race == 0 and self.gender == 1: #wm
                    self.fraud_algo(None)

                elif self.wealth > (self.p.appeal_wealth + 0.02) and self.race == 0 and self.gender == 0: #ww
                    # self.update_star()
                    self.fraud_algo(None)
                
                elif self.wealth > (self.p.appeal_wealth + 0.04) and self.race == 1 and self.gender == 1: #nwm
                    # self.update_star()
                    self.fraud_algo(None)

                elif self.wealth > (self.p.appeal_wealth +0.08) and self.race == 1 and self.gender == 0: #nww
                    self.fraud_algo(None)
            
            elif self.wealth > self.p.appeal_wealth:
                self.fraud_algo(None)




    def convict(self):
        """ Conviction and Consequences"""
        rng = np.random.default_rng()
        if self.fraud_pred == 1:
            if self.p.convict_inter:
                if self.race == 0 and self.gender == 1: #wm
                    self.wealth = np.clip(self.wealth - np.max([0.05,(self.wealth *0.1)]),0,1)

                elif self.race == 0 and self.gender == 0: #ww
                    self.wealth = np.clip(self.wealth - np.max([0.05,(self.wealth *0.1)]) - 0.02,0,1)
                
                elif self.race == 1 and self.gender == 1: #nwm
                    self.wealth = np.clip(self.wealth - np.max([0.05,(self.wealth *0.1)]) - 0.04,0,1)

                elif self.race == 1 and self.gender == 0: #nww
                    self.wealth = np.clip(self.wealth - np.max([0.05,(self.wealth *0.1)]) - 0.08,0,1)
            
            elif self.wealth > self.p.appeal_wealth:
                self.wealth = np.clip(self.wealth - np.max([0.05,(self.wealth *0.1)]),0,1)
            
            self.convicted =+ 1
        __, self.fraud, __ = fraud_val(self.wealth, self.race, self.gender, self.health,self.p.star_version,self.p.synth_data_acc, self.p.abm_eval, self.p.clf)
            # self.fraud = 0
    
    def wealth_grow(self):
        self.wealth = min(1,self.wealth+pow(self.wealth,2)*0.1)






            
     