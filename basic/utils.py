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





def fairness_metrics(data, pr = False):
        y_true = list(data.fraud)
        y_pred = list(data.fraud_pred)
        gender = data.gender
        race = data.race
        # print((y_pred), y_true)

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
                if pr:
                    print('dpd',temp_dpd)
                    print('eod',temp_eod)
                eod.append(temp_eod)


        # dpd = demographic_parity_difference( y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        if (sum(y_true) != 0 and sum(y_pred) != 0):
            if not pr:
                eod_gender = eod[0]
                eod_race = eod[1]
                dpd_gender = dpd[0]
                dpd_race = dpd[1]   
                eval_acc = 1 - sum(abs(np.array(y_true)-np.array(y_pred)))/len(y_true)
                # x = 1 - sum(abs(np.array(y_pred)-np.array(data.star)))/len(y_true)
                # print(self.eval_acc, x
                # )

                # print(eod_gender, eod_race, dpd_gender, dpd_race, eval_acc)

                return eod_gender, eod_race, dpd_gender, dpd_race, eval_acc









if __name__ == '__main__':

    classifier_train('clf_x.txt', 'clf_y.txt')
    