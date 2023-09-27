# imports
from agent import Person
from model import TaxFraudModel
from utils import transform_pd

# Model design
import agentpy as ap
import networkx as nx 
import random 
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython 

# other 
import pickle
import argparse

def run_model(clf,expi, star_version, synth_data_acc, abm_eval):

    parameters = {
        'my_parameter':42,
        'agents':1000,
        'steps':150,
        'acc': 0.9, # accuracy of fraud prdediction
        'conviction_rate': 1,
        'appeal_wealth': 0.28, # minimal wealth needed for appeal (could also become a param for distr. eventually)
        #'wealth_impact',
        'clf' : clf,
        'expi' : expi,
        'fraud_det': 0,
        'fairness_metrics' : True,
        'star_version': star_version,
        'synth_data_acc' : synth_data_acc, 
        'abm_eval' : abm_eval
        
    }
    
    exp1 = ap.Experiment(TaxFraudModel, parameters, iterations =50, record=True)
    results_baseline = exp1.run() 
    df_baseline = results_baseline['variables']['Person']
    df_baseline_mlp = transform_pd(df_baseline)

    return df_baseline




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", default= None, help = "classifier")
    parser.add_argument("-e", "--expi", metavar="IMAGE_FLIP", help = "experiment")
    parser.add_argument("-o", "--out", metavar="IMAGE_FLIP", help = "outputdir", default = '')
    parser.add_argument("-s", "--star", metavar="star version", help = "starversion", default = None)
    parser.add_argument("-a", "--abm_eval", metavar="abm eval", help = "starversion", default = None)
    parser.add_argument("-sda", "--synth_data_acc", metavar="synthetic data cc", help = "starversion", default = None, type = float)

    args = parser.parse_args()
    kwargs = vars(args)
    clf = kwargs.pop("clf")
    expi = kwargs.pop("expi")
    outdir = kwargs.pop("out")
    star_version = kwargs.pop("star")
    abm_eval = kwargs.pop("abm_eval")
    synth_data_acc = kwargs.pop("synth_data_acc")

    df = run_model(clf,expi, star_version, synth_data_acc, abm_eval)

    filename =  (outdir + '/df_{}_{}_{}_{}.pkl').format(clf,expi,star_version,abm_eval)

    with open(filename, "wb") as handle:
        pickle.dump(df, handle) 
        print('saved '+ filename)


     




