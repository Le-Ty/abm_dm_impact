#libs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import pickle


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



def classifier_train(X_name, y_name):

    X = (pickle.load(open(X_name, 'rb')))
    y = (pickle.load(open(y_name, 'rb')))


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    
    with open("clf.pkl", "wb") as f:
        pickle.dump(clf, f)




if __name__ == '__main__':

    classifier_train('clf_x.txt', 'clf_y.txt')