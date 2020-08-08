import xgboost as xgb 
import pandas as pd 
import numpy as np 

data=pd.read_csv('data_revised.csv',index_col=0)
features=data.drop(['label','year'],axis=1)
target=data.label

n=int(len(features)*0.8)
X_train=features[:n]
X_test=features[n:]
y_train=target[:n]
y_test=target[n:]

dtrain=xgb.DMatrix(X_train,y_train)
dtest=xgb.DMatrix(X_test,y_test)

param={'boosting_type':'gbdt',
        'max_depth': 2, 
        'eta': 1, 
        'objective': 'binary:logistic',
        'learning rate': 0.5,
        'nthread': 4,
        'eval_metric':'auc'}
evallist=[(dtest, 'eval'), (dtrain, 'train')]
#Training
bst=xgb.train(param,dtrain,num_boost_round=100,evals=evallist,early_stopping_rounds=15)
y_pred=bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
#Plot ROC Curve
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc
def Roc_curve(y_true,y_prediction):
        fpr,tpr,threshold=roc_curve(y_true,y_prediction)
        AUC=auc(fpr,tpr)
        plt.figure(figsize=(20,20))
        plt.xlim(0.0,1.0)
        plt.ylim(0.0,1.0)
        plt.plot(fpr,tpr,color='purple',lw=2,linestyle='--',label='ROC Area is: %3.2f'%AUC)
        plt.plot([0,1],[0,1],color='yellow',linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()
Roc_curve(y_test,y_pred)
#bst.save_model('XGBoost_model.txt')