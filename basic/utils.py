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


def viz(data, x, y, hue):
    fig, ax = plt.subplots()

    sns.lineplot(ax = ax,
                 data = data,
                 x = x,
                 y = y,
                 hue = hue,
                 marker = 'o')

    plt.show()



def transform_pd(df_baseline):

    df_baseline['misclassifications'] = (df_baseline['fraud_pred'] - df_baseline['fraud'])
    df_baseline[df_baseline['fraud_pred'] ==0]['wealth'].min()

    df_baseline.head()
    df_baseline = df_baseline[df_baseline.fraud_pred != -1]
    df_baseline[df_baseline['misclassifications'] == 0]

    df_baseline['intersect'] = list(df_baseline['gender'])
    mask1 = ((df_baseline['gender'] == 0) & (df_baseline['race'] == 0))
    df_baseline.loc[mask1, 'intersect'] = 'fw'
    mask2 = ((df_baseline['gender'] == 1) & (df_baseline['race'] == 0))
    df_baseline.loc[mask2, 'intersect'] = 'mw'

    mask3 = ((df_baseline['gender'] == 0) & (df_baseline['race'] == 1))
    df_baseline.loc[mask3, 'intersect'] = 'fnw'

    mask4 = ((df_baseline['gender'] == 1) & (df_baseline['race'] == 1))
    df_baseline.loc[mask4, 'intersect'] = 'mnw'

    return df_baseline


def delta_function(disc_axis, y_axis, df, df_baseline):
    """ Delta function visualizes absolute difference between baseline scenario and more complex scenario """

    if disc_axis == 'intersect':
        data = []
        for i in ['fnw', 'fw', 'mnw', 'mw']:
            df_b1 = df_baseline.iloc[(df_baseline[disc_axis] == i).values] 
            df_wb1 = df.iloc[(df[disc_axis] == i).values][y_axis] - df_b1.groupby(level='t').mean()[y_axis]
            df_x1 = df.iloc[(df[disc_axis] == i).values]
            df_x1[y_axis] = df_wb1
            data.append(df_x1)
        data = pd.concat(data)

    else:
        df_b1 = df_baseline.iloc[(df_baseline[disc_axis] == 1).values] 
        df_wb1 = df.iloc[(df[disc_axis] == 1).values][y_axis] - df_b1.groupby(level='t').mean()[y_axis]
        df_x1 = df.iloc[(df[disc_axis] == 1).values]
        df_x1[y_axis] = df_wb1
        
        df_b0 = df_baseline.iloc[(df_baseline[disc_axis] == 0).values] 
        df_wb0 = df.iloc[(df[disc_axis] == 0).values] [y_axis] - df_b0.groupby(level='t').mean()[y_axis]
        df_x0 = df.iloc[(df[disc_axis] == 0).values]
        df_x0[y_axis] = df_wb0
        data = pd.concat([df_x0, df_x1])
    
    return data



def non_compliance(tolerance =0.3):
    #weight requirements 
    non_compliance = weighted_resources + tolerance


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
    fairness_metrics(X_test)




    with open("star_is4.pkl", "wb") as f:
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

    if True: #mitigate == 'decorrelate':
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




    with open("clf.pkl", "wb") as f:
        pickle.dump(pipe, f)


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


        


def generate_init(star_version, star_acc, train_clf = True, n = 1, fraud_det = 0):


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
            fraud = fraud_val(wealth, fraud_det, False)
        
        else:
            fraud = fraud_val(wealth, fraud_det, True)
            star = generate_star(i, gender, wealth, health, fraud, star_version, star_acc)[0]

        
        if train_clf:
            x.append([i,gender,wealth,health, star])
            y.append(fraud)
        else:
            g.append(gender)
            w.append(wealth)
            h.append(health)
            f.append(fraud)
            s.append(star)





    fraud_pred = np.full(n,-1)[0]
    convicted = np.full(n,0)

    # print(f)

    if train_clf:
        return x,y
    else:
        return race[0],g[0],w[0],h[0],s[0], f[0],fraud_pred,convicted[0]




def plot_heatmap(df,y, target, title = 'correlation matrix'):
    df[target] = list(y)
    # df = df.rename(columns={"had_inpatient_days_True": "had_inpatient_days"})
    cols = list(df.columns)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(round(df.corr(), 2), cmap="coolwarm")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(cols)), labels=cols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                round(df.corr().to_numpy()[i, j], 2),
                ha="center",
                va="center",
            )

    ax.set_title(f"{title}")
    plt.show()





def fairness_metrics(data):

    y_true = list(data.fraud)
    y_pred = list(data.fraud_pred)
    gender = data.gender
    race = data.race

    
    for i in [gender,race]:

        temp_dpd = demographic_parity_ratio( y_true=y_true, y_pred=y_pred, sensitive_features=i)
        temp_eod = equalized_odds_ratio( y_true=y_true, y_pred=y_pred, sensitive_features=i)

        # print('dpd',temp_dpd)
        # print('eod',temp_eod)
    # dpd = demographic_parity_difference( y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)


def generate_star(race, gender, wealth, health, fraud, star_version, star_acc):

    rng = np.random.default_rng() 

    if star_version == 'uniform':
        star = rng.choice([fraud, 1- fraud], 1, p = [star_acc, 1- star_acc])

    elif star_version == 'qualitative':
        star_prob = rng.binomial(1,((wealth-0.2)**3+0.6)*(((-x+0.6)*1.6)**3+0.4))
        star =  np.random.choice([fraud,1 - fraud], 1, p =[fraud_det, 1- fraud_det])



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
    else:
        raise ValueError('generate_star did not get the correct variable, check the paramters of the ABM')

    
    return star







if __name__ == '__main__':

    classifier_train('clf_x.txt', 'clf_y.txt')
    