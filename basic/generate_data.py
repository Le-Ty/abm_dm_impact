
#libs
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from fairlearn.reductions import DemographicParity
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import random
import os


def fraud_val(wealth, fraud_det = 0, star = False):
    # make fraud dependent on wealth 
    rng = np.random.default_rng() 

 
    if wealth > 0.40823212:
        p_det = 1
    else: 
        p_det = 0
    
    if not star:      
        p_prob = rng.binomial(1,np.clip(((wealth-0.75)**4+0.3), 0,0.9))
        return np.random.choice([p_det,p_prob], 1, p =[fraud_det, 1- fraud_det])[0]  

    else:
        p_prob = rng.binomial(1,np.clip(((wealth-0.2)**3+0.3), 0,0.9))
        return np.random.choice([p_det,p_prob], 1, p =[fraud_det, 1- fraud_det])[0]  


        


def generate_init(star_version, synth_data_acc, abm_eval, train_clf = True, n = 1, fraud_det = 0):


    np.random.seed(42)
    

    dir1 = ("/gpfs/home4/ltiyavorabu/abm/basic/data/distributions_init.pickle")
    dir2 = ("/gpfs/home4/ltiyavorabu/abm/basic/data/values_init.pickle")

    local1 = ("data/distributions_init.pickle")
    local2 = ("data/values_init.pickle")

    with open(dir1, "rb") as f:
        d_fnw = pickle.load(f)
        d_mw = pickle.load(f)
        d_mnw = pickle.load(f)
        d_fw = pickle.load(f)
    with open(dir2, "rb") as f:
        v_fnw = pickle.load(f)
        v_mw = pickle.load(f)
        v_mnw = pickle.load(f)
        v_fw = pickle.load(f)

    rng = np.random.default_rng()        
    
    # race
    y = []
    x = []
    g = []
    w = []
    h = []
    f =[]
    s = []
    race =  rng.binomial(1,0.2,n) #binary not white0.2 /  white for the moment 0.8
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
        
        

        if star_version == None:
            fraud_train = fraud_val(wealth, fraud_det, False)
        
        else:
            gt_fraud = fraud_val(wealth, fraud_det, True)
            
            if abm_eval == 'GT':
                abm_eval_fraud = gt_fraud
                fraud_train = generate_bias(i, gender, wealth, health, gt_fraud, star_version)[0]
                star = rng.choice([fraud_train, 1- fraud_train], 1, p = [synth_data_acc, 1- synth_data_acc])[0]

            elif abm_eval == "HIST":
                abm_eval_fraud = generate_bias(i, gender, wealth, health, gt_fraud, star_version = 'hist')[0]
                fraud_train = generate_bias(i, gender, wealth, health, gt_fraud, star_version)[0]
                star = rng.choice([fraud_train, 1- fraud_train], 1, p = [synth_data_acc, 1- synth_data_acc])[0]

            
            else:
                fraud_train = abm_eval_fraud = generate_bias(i, gender, wealth, health, gt_fraud, star_version)[0] 
                star =  rng.choice([fraud_train, 1- fraud_train], 1, p = [synth_data_acc, 1- synth_data_acc])[0]
        

            

        
        if train_clf:
            x.append([i,gender,wealth,health, star])
            y.append(fraud_train)
        else:
            g.append(gender)
            w.append(wealth)
            h.append(health)
            f.append(abm_eval_fraud)
            s.append(star)


    fraud_pred = np.full(n,-1)[0]
    convicted = np.full(n,0)


    if train_clf:
        return x,y
    else:
        return race[0],g[0],w[0],h[0],s[0], f[0],fraud_pred,convicted[0]




def generate_bias(race, gender, wealth, health, fraud, star_version, star_acc = 0.8):

    rng = np.random.default_rng() 

    if star_version == 'uniform':
        star = rng.choice([fraud, 1- fraud], 1, p = [star_acc, 1- star_acc])

    # elif star_version == 'qualitative':
    #     star_prob = rng.binomial(1,((wealth-0.2)**3+0.6)*(((-x+0.6)*1.6)**3+0.4))
    #     star =  np.random.choice([fraud, 1 - fraud], 1, p =[fraud_det, 1- fraud_det])

    elif star_version == 'is1':
        if gender == 0: #women
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc + 0.1, 1- star_acc - 0.1])
        else: # men
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc - 0.1, 1- star_acc + 0.1])

    elif star_version == 'is2':
        # gender & race
        if (gender == 0 and race == 1).all():
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc + 0.15, 1- star_acc - 0.15])
        else:
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc - 0.025, 1- star_acc + 0.025])

    elif star_version == 'is3':
        if (gender == 0 and race == 1 and wealth < 0.2).all():
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc + 0.15, 1- star_acc - 0.15])
        else:
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc - 0.025, 1- star_acc + 0.025])

    elif star_version == 'is4':
        if (gender == 0 and race == 1 and wealth < 0.2 and health < 0.4).all():
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc + 0.15, 1- star_acc - 0.15])
        else:
            star = rng.choice([fraud, 1- fraud], 1, p = [star_acc - 0.0125, 1- star_acc + 0.0125])

    elif star_version == 'hist':
        
         star = rng.binomial(1, ((wealth-0.2)**3+0.3)*((wealth+0.3)**(-1)+(-0.7)),1)

    else:
        raise ValueError('generate_star did not get the correct variable, check the admissible paramters of the ABM')

    
    return star
