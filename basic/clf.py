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

from utils import plot_heatmap
from utils import fairness_metrics





def classifier_train_star(X, y, mitigate = 'None', viz = False):

    # X = (pickle.load(open(X_name, 'rb')))
    # y = (pickle.load(open(y_name, 'rb')))

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    y=y.rename(columns = {0:'y'})
    X = X.rename(columns = {0: 'race', 1:'gender', 2:'wealth', 3:'health', 4: 'star'})
    df = pd.concat([X,y], axis =1)


    # # Separate majority and minority classes
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

    # X = df_baseline[['fraud','wealth', 'gender', 'race']]


    y = df_up_down_sampled['y']
    X = df_up_down_sampled.drop('y', axis = 1)
    X = X.rename(columns = {0: 'race', 1:'gender', 2:'wealth', 3:'health', 4:'star'})

    if True: #mitigate == 'decorrelate':
        cr = CorrelationRemover(sensitive_feature_ids=['race'])
        # cr.fit(X)
        X_t = cr.fit_transform(X)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.rename(columns = {0:'gender', 1:'wealth', 2:'health', 3:'star'})
        X_t.insert(0,'race',list(X['race']))
        # X_t.insert(0,'gender',list(X['gender']))
        if viz:
            plot_heatmap(pd.DataFrame(X),X['race'],target = 'race', title= "Correlation values in the original dataset")

            plot_heatmap(pd.DataFrame(X_t),X['race'], target = 'race', title="Correlation values in the decorrelated dataset")

        cr2 = CorrelationRemover(sensitive_feature_ids=['gender'])
        X_t2 = cr2.fit_transform(X_t)
        X_t2 = pd.DataFrame(X_t2)
        X_t2 = X_t2.rename(columns = {0:'race', 1:'wealth', 2:'health', 3: 'star'})
        X_t2.insert(1,'gender',list(X['gender']))
        # X_t.insert(0,'gender',list(X['gender']))

        if viz:
            plot_heatmap(pd.DataFrame(X),X['gender'], target = 'gender', title="Correlation values in the original dataset")

            plot_heatmap(pd.DataFrame(X_t2),X['gender'], target = 'gender',  title="Correlation values in the decorrelated dataset")
        
        
        X = X_t2


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    if mitigate == 'reduce': 
        scaler = StandardScaler()
        model = scaler.fit(X)
        X_train = pd.DataFrame(model.transform(X_train), columns = ['race', 'gender', 'wealth', 'health', 'star'])
        X_test = pd.DataFrame(model.transform(X_test), columns = ['race', 'gender', 'wealth', 'health', 'star'])
        pipe =  ExponentiatedGradient(SGDClassifier(loss = 'log_loss', penalty = 'elasticnet', alpha = 0.01), constraints=DemographicParity(),eps=0.1)
        pipe.fit(X_train, pd.DataFrame(y_train), sensitive_features= X_train['race'])
        y_pred = pipe.predict(X_test)
        print((len(y_pred) -abs(y_pred - np.array(y_test).flatten()).sum())/len(y_pred))
    
    elif mitigate == 'adv':

        pipe = AdversarialFairnessClassifier(
            backend="torch",
            predictor_model=[50, "leaky_relu"],
            adversary_model=[3, "leaky_relu"],
            batch_size=2 ** 8,
            progress_updates=0.5,
            random_state=123,
        )

        pipe.fit(X_train, y_train, sensitive_features=X_train['race'])
        y_pred = pipe.predict(X_test)
        print((len(y_pred) -abs(y_pred - np.array(y_test).flatten()).sum())/len(y_pred))
        print(pipe)
        # print(AdversarialFairnessClassifier.predictor_model)
 



        
    else:
        # pipe = make_pipeline(StandardScaler(), MLPClassifier(solver='adam', alpha = 0.005, hidden_layer_sizes=(40, 20), random_state=1))
        pipe = make_pipeline(StandardScaler(), SGDClassifier(loss = 'log_loss', penalty = 'l1', alpha = 0.01)) # BaggingClassifier(estimator=SVC(class_weight={0:0.50, 1:0.50}),n_estimators=10, random_state=0))
        pipe.fit(X_train, y_train) 
        y_pred = pipe.predict(X_test)
        print(pipe.score(X_test,y_test))
    # clf = RandomForestClassifier(n_estimators=500)
    # clf.fit(X_train, y_train) 
    # score = cross_val_score(pipe, X_train, y_train, cv=cv)

    X_test['fraud_pred'] = y_pred
    # print(X_test)
    X_test['fraud'] = list(y_test)
    # print(X_test)
    fairness_metrics(X_test, True)




    with open("s_is4_dec.pkl", "wb") as f:
        pickle.dump(pipe, f)

def classifier_train(X, y, mitigate = 'None', viz = False):

    # X = (pickle.load(open(X_name, 'rb')))
    # y = (pickle.load(open(y_name, 'rb')))

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    y=y.rename(columns = {0:'y'})
    X = X.rename(columns = {0: 'race', 1:'gender', 2:'wealth', 3:'health', 4:'star'})
    df = pd.concat([X,y], axis =1)


    # # Separate majority and minority classes
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

    # X = df_baseline[['fraud','wealth', 'gender', 'race']]


    y = df_up_down_sampled['y']
    X = df_up_down_sampled.drop('y', axis = 1)
    X = X.rename(columns = {0: 'race', 1:'gender', 2:'wealth', 3:'health'})

    if False: #mitigate == 'decorrelate':
        cr = CorrelationRemover(sensitive_feature_ids=['race'])
        # cr.fit(X)
        X_t = cr.fit_transform(X)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.rename(columns = {0:'gender', 1:'wealth', 2:'health'})
        X_t.insert(0,'race',list(X['race']))
        # X_t.insert(0,'gender',list(X['gender']))
        if viz:
            plot_heatmap(pd.DataFrame(X),X['race'],target = 'race', title= "Correlation values in the original dataset")

            plot_heatmap(pd.DataFrame(X_t),X['race'], target = 'race', title="Correlation values in the decorrelated dataset")

        cr2 = CorrelationRemover(sensitive_feature_ids=['gender'])
        X_t2 = cr2.fit_transform(X_t)
        X_t2 = pd.DataFrame(X_t2)
        X_t2 = X_t2.rename(columns = {0:'race', 1:'wealth', 2:'health'})
        X_t2.insert(1,'gender',list(X['gender']))
        # X_t.insert(0,'gender',list(X['gender']))

        if viz:
            plot_heatmap(pd.DataFrame(X),X['gender'], target = 'gender', title="Correlation values in the original dataset")

            plot_heatmap(pd.DataFrame(X_t2),X['gender'], target = 'gender',  title="Correlation values in the decorrelated dataset")
        
        
        X = X_t2


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    if mitigate == 'reduce': 
        scaler = StandardScaler()
        model = scaler.fit(X)
        X_train = pd.DataFrame(model.transform(X_train), columns = ['race', 'gender', 'wealth', 'health'])
        X_test = pd.DataFrame(model.transform(X_test), columns = ['race', 'gender', 'wealth', 'health'])
        pipe =  ExponentiatedGradient(SGDClassifier(loss = 'log_loss', penalty = 'elasticnet', alpha = 0.01), constraints=DemographicParity(),eps=0.1)
        pipe.fit(X_train, pd.DataFrame(y_train), sensitive_features= X_train['race'])
        y_pred = pipe.predict(X_test)
        # print((len(y_pred) -abs(y_pred - np.array(y_test).flatten()).sum())/len(y_pred))
    
    elif mitigate == 'adv':

        pipe = AdversarialFairnessClassifier(
            backend="torch",
            predictor_model=[50, "leaky_relu"],
            adversary_model=[3, "leaky_relu"],
            batch_size=2 ** 8,
            progress_updates=0.5,
            random_state=123,
        )

        pipe.fit(X_train, y_train, sensitive_features=X_train['race'])
        y_pred = pipe.predict(X_test)
        # print((len(y_pred) -abs(y_pred - np.array(y_test).flatten()).sum())/len(y_pred))
        # print(pipe)
        # print(AdversarialFairnessClassifier.predictor_model)
 



        
    else:
        # pipe = make_pipeline(StandardScaler(), MLPClassifier(solver='adam', alpha = 0.005, hidden_layer_sizes=(40, 20), random_state=1))
        pipe = make_pipeline(StandardScaler(), SGDClassifier(loss = 'log_loss', penalty = 'l1', alpha = 0.01)) # BaggingClassifier(estimator=SVC(class_weight={0:0.50, 1:0.50}),n_estimators=10, random_state=0))
        pipe.fit(X_train, y_train) 
        y_pred = pipe.predict(X_test)
        print('SCORE' + pipe.score(X_test,y_test))
    # clf = RandomForestClassifier(n_estimators=500)
    # clf.fit(X_train, y_train)
    # score = cross_val_score(pipe, X_train, y_train, cv=cv)

    X_test['fraud_pred'] = y_pred
    # print(X_test)
    X_test['fraud'] = list(y_test)
    # print(X_test)
    fairness_metrics(X_test)




    with open("s_is3_08.pkl", "wb") as f:
        pickle.dump(pipe, f)
