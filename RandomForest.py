import pandas as pd 
import numpy as np 
data=pd.read_csv('data_revised.csv',index_col=0)

features=data.drop(['label','year'],axis=1)
target=data.label

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve,auc
#Given n-percentage to find random training set and testing set

n=int(len(features)*0.8)
X_train=features[:n]
y_train=target[:n]
X_test=features[n:]
y_test=target[n:]

#RandomForest
r=RandomForestClassifier(n_estimators=20,min_samples_split=5,max_depth=10)
R=r.fit(X_train,y_train)
#Make prediction of probability for 0 and 1 
y_pred_proba=R.predict_proba(X_test)
importance=R.feature_importances_
indices=np.argsort(importance)[::-1] #Get the importance of features
bb=features.columns
for f in range(features.shape[1]):
    print(("%2d) %-*s %f"%(f + 1, 30, bb[f], importance[indices[f]])))
#Drop low corrleation feature TOT_ASSET_LABEL
features_new=features.drop(['TOT_PROFIT_label'],axis=1)
X_new_train=features_new[:n]
X_new_test=features_new[n:] #Define y_train and y_test before
R_new=RandomForestClassifier(n_estimators=20,min_samples_split=5,max_depth=10).fit(X_new_train,y_train)
y_new_predict_proba=R_new.predict_proba(X_new_test)
#Plot ROC Curve
fpr,tpr,threshold=roc_curve(y_test,y_new_predict_proba[:,1])
AUC=auc(fpr,tpr)
lw=2
plt.figure(figsize=(20,20))
plt.plot([0,1],[0,1],color='darkblue',lw=lw,linestyle='--')
plt.plot(fpr,tpr,color='purple',lw=lw,label='ROC Area is %.2f'%AUC)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('Random Forest ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='upper right')
plt.show()