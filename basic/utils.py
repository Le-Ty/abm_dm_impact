#libs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import random


def viz(data, x, y, hue):
    fig, ax = plt.subplots()

    sns.lineplot(ax = ax,
                 data = data,
                 x = x,
                 y = y,
                 hue = hue,
                 marker = 'o')

    plt.show()




def delta_function(disc_axis, y_axis, df, df_baseline):
    """ Delta function visualizes absolute difference between baseline scenario and more complex scenario """

    df_b1 = df_baseline.iloc[(df_baseline[disc_axis] == 1).values] 
    df_wb1 = df.iloc[(df[disc_axis] == 1).values][y_axis] - df_b1.groupby(level='t').mean()[y_axis]
    df_x1 = df.iloc[(df[disc_axis] == 1).values]
    df_x1[y_axis] = df_wb1
    
    df_b0 = df_baseline.iloc[(df_baseline[disc_axis] == 0).values] 
    df_wb0 = df.iloc[(df[disc_axis] == 0).values] [y_axis] - df_b0.groupby(level='t').mean()[y_axis]
    df_x0 = df.iloc[(df[disc_axis] == 0).values]
    df_x0[y_axis] = df_wb0
    
    return pd.concat([df_x0, df_x1])



def classifier_train(X, y):

    # X = (pickle.load(open(X_name, 'rb')))
    # y = (pickle.load(open(y_name, 'rb')))


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    
    with open("clf.pkl", "wb") as f:
        pickle.dump(clf, f)


def generate_init(train_clf = True, n = 1):

    np.random.seed(42)

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
    g = []
    w = []
    h = []
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
        
        if train_clf:
            x.append([i,gender,wealth,health])
        else:
            g.append(gender)
            w.append(wealth)
            h.append(health)


    # fraud
    fraud = rng.binomial(1,0.5,n)
    fraud_pred = np.full(n,-1)
    convicted = np.full(n,0)

    if train_clf:
        return x,fraud
    else:
        return race[0],g[0],w[0],h[0],fraud,fraud_pred,convicted








if __name__ == '__main__':

    classifier_train('clf_x.txt', 'clf_y.txt')
    