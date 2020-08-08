import pandas as pd 
data=pd.read_csv('data_revised.csv',index_col=0)

import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
features=data.drop(['label','year'],axis=1)
response=data.label
#Basic Decision Tree Model
def DTClass(fea,res):
    from sklearn.metrics import roc_curve,auc
    c=DecisionTreeClassifier(min_samples_split=10,max_depth=50,random_state=30)
    n=int(len(fea)*0.8)
    X_train=fea[:n]
    y_train=res[:n]
    X_test=fea[n:]
    y_test=res[n:]
    result=c.fit(X_train,y_train)
    y_pred_proba=result.predict_proba(X_test)
    #Plot RoC Curve
    fpr,tpr,threshold=roc_curve(y_test,y_pred_proba[:,1])
    AUC=auc(fpr,tpr)
    plt.figure(figsize=(20,20))
    plt.plot([0,1],[0,1],color='darkblue',lw=2,linestyle='--') #Random Guessing
    plt.plot(fpr,tpr,color='yellow',lw=2,linestyle='dashed',label='ROC Area is %.2f'%AUC)
    plt.title('DecisionTree ROC&AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='upper right')
    plt.show()

DTClass(features,response)

