import pandas as pd 
import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve,auc
data=pd.read_csv('data_revised.csv',index_col=0)
import matplotlib.pyplot as plt 

features=data.drop(['label','year'],axis=1)
target=data.label
#Define the training set and testing set 
n=int(len(features)*0.8)
X_train=features[:n]
X_test=features[n:]
y_train=target[:n]
y_test=target[n:]

best=[]
for i in range(1,21):
    g=GradientBoostingClassifier(n_estimators=20,max_depth=i+1,learning_rate=0.5).fit(X_train,y_train)
    scores=g.score(X_train,y_train)
    a=best.append(scores)

aa=best.index(max(best))
#Use optimal score to fit GBDT model
G=GradientBoostingClassifier(n_estimators=20,max_depth=aa,learning_rate=0.5).fit(X_train,y_train)
y_pred_proba=G.predict_proba(X_test)
#Get the importance of features
importance=G.feature_importances_
indices=np.argsort(importance)[::-1] #Get the importance of features
bb=features.columns
for f in range(features.shape[1]):
    print(("%2d) %-*s %f"%(f + 1, 30, bb[f], importance[indices[f]])))
#Drop some less importance features
feature_new=features.drop(['TOT_SIZE_ASSETS','TOT_PROFIT_label','Revenue'],axis=1)
X_new_train=feature_new[:n]
X_new_test=feature_new[n:]
G_new=GradientBoostingClassifier(n_estimators=20,max_depth=aa,learning_rate=0.5).fit(X_new_train,y_train)
y_new_pred_proba=G_new.predict_proba(X_new_test)
#Plot ROC Curve
fpr,tpr,threshold=roc_curve(y_test,y_pred_proba[:,1])
AUC=auc(fpr,tpr)
lw=2
plt.figure(figsize=(20,20))
plt.plot([0,1],[0,1],color='darkblue',lw=lw,linestyle='--')
plt.plot(fpr,tpr,color='purple',lw=lw,label='ROC Area is %.2f'%AUC)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='upper right')
plt.show()
