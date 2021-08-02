from numpy.lib.function_base import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import load_dicts as ld
import pickle
import sys
from scipy import stats
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

def main():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'AMOHData'

    df = pd.read_csv(fm+'_churn.csv')
    #feature_cols = ['time mean','total transactions','monthly average','monthly spending regression'] 
    #feature_cols = ['time mean','total transactions','total product wkend','monthly spending regression','monthly transaction regression']# best so far
    feature_cols = ['time max','monthly average','total transactions','monthly spending regression','monthly transaction regression']
    #feature_cols = ['total product wkday','total transactions wkend','maximum product wkend','monthly spending regression','monthly transaction regression']
    #feature_cols = ['average amount 2','average discount 2','percent discount 2','monthly transaction regression']
    #feature_cols = ['monthly transaction regression']

    #feature_cols = ['time mean', 'time max', 'monthly average', 'total transactions', 'maximum amount', 'total transactions wkday', 
        #'minimum discount wkday', 'monthly spending regression', 'monthly transaction regression']

    feature_cols = ['monthly transaction regression']

    X = df.loc[:,feature_cols].to_numpy()
    y = df.left.to_numpy()

    scores = {} 
    model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(random_state=0),GaussianNB(),KNeighborsClassifier(),SVC(),SGDClassifier(),MLPClassifier()]

    model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(random_state=0)]

    X_features = {}
    #for model in model_list:
    #    X_features[str(model)] = select_feat(model)
    

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        for model in model_list:
            #X = df.loc[:,X_features[str(model)]].to_numpy()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = roc_auc_score(y_test,y_pred)
            #score = len(y_test[y_pred==y_test])/len(y_test)
            if str(model) in scores:
                scores[str(model)].append(score)
            else:
                scores[str(model)] = [score]

    
    for model in X_features:
        print(f'{model:25}',end=' ')
        print(X_features[model])
    
    print(' ')

    for score in scores:
        print(f'{score:40}',end=' ')
        print(sum(scores[score])/len(scores[score]))

def single_feature(feature='time max'):
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'AMOHData'

    df = pd.read_csv(fm+'_churn.csv')
    feature_cols = ['time max','monthly average','total transactions','monthly spending regression','monthly transaction regression']


    X = df.loc[:,feature].to_numpy().reshape(-1,1)
    y = df.left.to_numpy().ravel()

    scores = {} 
    model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(random_state=0),GaussianNB(),KNeighborsClassifier(),SVC(),SGDClassifier(),MLPClassifier()]

    model_list = [LogisticRegression()]

    X_features = {}
  

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        for model in model_list:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = roc_auc_score(y_test,y_pred)
            if str(model) in scores:
                scores[str(model)].append(score)
            else:
                scores[str(model)] = [score]

    
    for model in X_features:
        print(f'{model:25}',end=' ')
        print(X_features[model])
    
    print(LogisticRegression())

    for score in scores:
        print(f'{feature:30}',end=' ')
        print(sum(scores[score])/len(scores[score]))
    print(' ')

def test_main():
    train = pd.read_csv('titanic_train.csv')

    feature_cols = ['Pclass','Parch']
    X = train.loc[:,feature_cols]

    y = train.Survived

    logreg = LogisticRegression()
    logreg.fit(X,y)

    test = pd.read_csv('titanic_test.csv')
    X_new = test.loc[:,feature_cols]
    new_pred_class = logreg.predict(X_new)
    kaggle_data = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':new_pred_class}).set_index('PassengerId')
    print(kaggle_data)

def left_graph():
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'AMOHData'

    df = pd.read_csv(fm+'_churn.csv')
    #feature_cols = ['total transactions','time max','time last','time mean']#,'monthly average','maximum product','time min','total product']
    feature_cols = ['time mean','total transactions','monthly average']
    left_df = df[df['left'] == 1]
    stay_df = df[df['left'] == 0]

    xr = left_df.loc[:,'total transactions'].to_numpy()
    yr = left_df.loc[:,'monthly spending regression'].to_numpy()
    zr = left_df.loc[:,'monthly transaction regression'].to_numpy()

    xb = stay_df.loc[:,'total transactions'].to_numpy()
    yb = stay_df.loc[:,'monthly spending regression'].to_numpy()
    zb = stay_df.loc[:,'monthly transaction regression'].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xb,zb,yb,c='b')
    ax.scatter(xr,zr,yr,c='r')
    
    plt.show()

def feature_importance_random_forest(feature_cols = None):
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'AMOHData'

    df = pd.read_csv(fm+'_churn.csv')
    
    if feature_cols == None:
        feature_cols = ['time max','monthly average','total transactions','monthly spending regression','monthly transaction regression']
        feature_cols = list(df.keys())
        feature_cols.pop(feature_cols.index('left'))


    X = df.loc[:,feature_cols].to_numpy()
    y = df.left.to_numpy()

    scores = {} 
    model = RandomForestClassifier()


    X_features = {}

    importances = []

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        #X = df.loc[:,X_features[str(model)]].to_numpy()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        score = roc_auc_score(y_test,y_pred)

        if len(importances) == 0:
            importances = model.feature_importances_
            #importances = model.coef_[0]
        else:
            importances += model.feature_importances_
            #importances += model.coef_[0]
            

        if str(model) in scores:
            scores[str(model)].append(score)
        else:
            scores[str(model)] = [score]
    
    for mod in X_features:
        print(f'{mod:25}',end=' ')
        print(X_features[mod])

    for score in scores:
        print(f'{score:40}',end=' ')
        print(sum(scores[score])/len(scores[score]))
    
    importances = importances/10

    imp_df = pd.DataFrame()
    imp_df['features'] = feature_cols
    imp_df['importance'] = importances

    print(imp_df.sort_values(by=['importance']))

    #for i in range(len(importances)):
    #    print(f'{feature_cols[i]:30}',end=' ')
    #    perc = importances[i]*100
    #    print(f'{perc:2.1f}%')
        

def select_feat(model,num_features = 5):
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'AMOHData'

    df = pd.read_csv(fm+'_churn.csv')

    X = list(df.keys())
    X.pop(X.index('left'))
    #X = ['time max','monthly average','total transactions','monthly spending regression','monthly transaction regression']
    #X = ['time max', 'monthly average', 'total transactions',  
         #'monthly spending regression', 'monthly transaction regression']

    #X = ['time max','monthly average','total transactions','monthly spending regression','monthly transaction regression']

    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:,X],
        df.left.to_numpy(),
        test_size=0.25,
        random_state=42)
     
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    sfs1 = sfs(model,k_features=num_features,forward=True,scoring='accuracy')
    sfs1 = sfs1.fit(X_train,y_train)

    feat_ind_cols = list(sfs1.k_feature_idx_)
    feat_cols = []
    for i in feat_ind_cols:
        feat_cols.append(X[i])

    return feat_cols

    print(feat_cols)
    for i in feat_cols:
        print(X[i])
        


if __name__ == "__main__":
    start = datetime.now()
    print(start)

    #main()

    if 1==1:
        
        single_feature('time max')
        single_feature('monthly average')
        single_feature('total transactions')
        single_feature('monthly spending regression')
        single_feature('monthly transaction regression')

    #feature_importance_random_forest(select_feat(RandomForestClassifier(random_state=0)))
    #print(select_feat(RandomForestClassifier()))
    #feature_importance_random_forest()

    end = datetime.now()-start
    print(f"{end.seconds} seconds")




    #feature_cols = ['time mean','total transactions','monthly average','monthly transaction regression','monthly transaction regression']
    #feature_cols = ['monthly average','total transactions','monthly spending regression']
    #feature_cols = ['time mean','maximum product','total transactions','average discount','average amount wkday'] # sfs svc
    #feature_cols = ['total transactions','time max','time last','time mean']#,'monthly average','maximum product','time min','total product']
    #feature_cols = ['time mean','monthly average','total transactions','total product','minimum product wkday'] # sfs logreg
    #feature_cols = ['time mean','total transactions','total discount wkday','average discount','average product wkday'] # sfs KNeighborsClassifier
    #feature_cols = ['time mean','total transactions','minimum discount','minimum product wkday','percent discount wkday'] # DecisionTree
    #feature_cols = ['time mean','time max','monthly average','total transactions','maximum discount','percent discount','average amount wkday','total transactions wkend']