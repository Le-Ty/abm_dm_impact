#libs
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils import resample

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

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    y=y.rename(columns = {0:'y'})
    df = pd.concat([X,y], axis =1)


    # Separate majority and minority classes
    df_majority = df[df['y'] ==0]
    df_minority = df[df['y'] ==1]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                    replace=False,    
                                    n_samples=4000)#Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,     
                                    n_samples=4000)# Combine minority class with downsampled majority class
    df_up_down_sampled = pd.concat([df_majority_downsampled, df_minority_upsampled])



    cv = KFold(n_splits=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    pipe = make_pipeline(StandardScaler(),  BaggingClassifier(estimator=SVC(class_weight={0:0.50, 1:0.50}),n_estimators=10, random_state=0))
    pipe.fit(X_train, y_train) 

    # clf = RandomForestClassifier(n_estimators=500)
    # clf.fit(X_train, y_train)
    # score = cross_val_score(pipe, X_train, y_train, cv=cv)
    print(pipe.score(X_test,y_test))

    with open("clf.pkl", "wb") as f:
        pickle.dump(pipe, f)


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
    y = []
    x = []
    g = []
    w = []
    h = []
    f =[]
    race =  rng.binomial(1,0.2,n) #binary not white0.2 /  white for the moment 0.8
    for i in race:
        gender = rng.binomial(1,0.5,1)[0]
        health = np.random.uniform(0,1,1)[0]

        if (gender == 0 and i == 1).all():
            wealth = (random.choices(v_fnw, weights=d_fnw, k=1))[0]
            fraud = rng.binomial(1,np.clip(((wealth-0.75)**4+0.3), 0,0.9))
        elif (gender == 1 and i == 1).all():
            wealth = (random.choices(v_mnw, weights=d_mnw, k=1))[0]
            fraud = rng.binomial(1,np.clip(((wealth-0.75)**4+0.3), 0,0.9))
        elif (gender == 1 and i == 0).all():
            wealth = (random.choices(v_mw, weights=d_mw, k=1))[0]
            fraud = rng.binomial(1,np.clip(((wealth-0.75)**4+0.3), 0,0.9))
        elif (gender == 0 and i == 0).all():
            wealth = (random.choices(v_fw, weights=d_fw, k=1))[0]
            fraud = rng.binomial(1,np.clip(((wealth-0.75)**4+0.3), 0,0.9))
        
        if train_clf:
            x.append([i,gender,wealth,health])
            y.append(fraud)
        else:
            g.append(gender)
            w.append(wealth)
            h.append(health)
            f.append(fraud)




    fraud_pred = np.full(n,-1)[0]
    convicted = np.full(n,0)

    if train_clf:
        return x,y
    else:
        return race[0],g[0],w[0],h[0],f[0],fraud_pred,convicted[0]








if __name__ == '__main__':

    classifier_train('clf_x.txt', 'clf_y.txt')
    