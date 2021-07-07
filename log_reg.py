import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import load_dicts as ld
import pickle
import sys
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def main():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'AMOHData'

    df = pd.read_csv(fm+'_churn.csv')
    #feature_cols = ['total transactions','time max','time last','time mean']#,'monthly average','maximum product','time min','total product']
    feature_cols = ['total transactions','monthly average','time mean']
    X = df.loc[:,feature_cols].to_numpy()
    y = df.left.to_numpy()

    scores = {} 
    model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GaussianNB()]

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        logreg = LogisticRegression()
        for model in model_list:
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = roc_auc_score(y_test,y_pred)
            if str(model) in scores:
                scores[str(model)].append(score)
            else:
                scores[str(model)] = [score]

        #logreg.fit(X_train,y_train)
        #y_pred = logreg.predict(X_test)
        #score = roc_auc_score(y_test,y_pred)
        
        
        #total.append(logreg.score(X_test,y_test))
        #print(logreg.score(X_test,y_test))
        #y_predict = logreg.predict(X_test)
        #total.append(len(y_test[y_predict==y_test])/len(y_test))
    
    for score in scores:
        print(f'{score:25}',end=' ')
        print(sum(scores[score])/len(scores[score]))

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


if __name__ == "__main__":
    start = datetime.now()

    main()

    end = datetime.now()-start
    print(f"{end.seconds} seconds")
