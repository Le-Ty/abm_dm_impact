
# imports
from agent import Person
from model import VirusModel, VirusModel_baseline

# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython 

# other 
import pickle
import argparse

def run_model(clf,expi):

    parameters = {
        'my_parameter':42,
        'agents':1000,
        'steps':50,
        'wealth_appeal_corr': 0, # >0 more wealth higher appeal chance
        'acc': 0.8, # accuracy of fraud prdediction
        'conviction_rate': 1,
        'appeal_wealth': 0.2, # minimal wealth needed for appeal (could also become a param for distr. eventually)
        #'wealth_impact',
        'clf' : clf,
        'expi' : expi,
        'fraud_det': 0,
        'fairness_metrics' : True
        
    }
    
    exp1 = ap.Experiment(VirusModel_baseline, parameters, iterations =2, record=True)
    results_baseline = exp1.run() 
    df_baseline = results_baseline['variables']['Person']
    df_baseline_mlp = transform_pd(df_baseline)

    return df_baseline





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", default="None", help = "classifier")
    parser.add_argument("-e", "--expi", metavar="IMAGE_FLIP", help = "experiment")

    args = parser.parse_args()
    kwargs = vars(args)
    clf = kwargs.pop("clf")
    expi = kwargs.pop("expi")

    df = run_model(clf,expi)

    filename =  ('df_ {} _{}').format(clf,expi)

    with open(filename, "wb") as f:
        pickle.dump(df) 


     




