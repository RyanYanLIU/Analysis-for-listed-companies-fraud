import pandas as pd 
import math
data=pd.read_csv('data_revised.csv',index_col=0)

import statsmodels.api as sm 
X=data.drop(['label','year'],axis=1)# X DataFrame
y=data.label #y Series
#建立逻辑回归模型，进行X自变量的训练和预测，以及y响应变量的联系
logit=sm.Logit(y,X).fit()
y_pred_old=logit.predict(X)
print(logit.summary())
#创建新的features，剔除某些不显著因子
X_new=X.drop(['SGAI','TATA','AQI','DEPI'],axis=1)
X_new=sm.add_constant(X_new)
logit_new=sm.Logit(y,X_new).fit()
y_pred=logit_new.predict(X_new)
#Confusion Matrix
#测试值为y，模型预测值为y_pred
def Confusion_Matrix(y_test,y_prediction):
    TruePositive=sum(y_pred[y==1])#取预测为1的，并且真实为1的样本数量总和
    FalsePositive=sum(1-y_pred[y==1]) 
    TrueNegative=sum(1-y_pred[y==0])
    FalseNegative=sum(y_pred[y==0])
    Precision=TruePositive/(TruePositive+FalsePositive)
    Recall=TruePositive/(TruePositive+FalseNegative)
    print('Precision: {}'.format(Precision))
    print('Recall: {}'.format(Recall))
    _confusion_matrix = pd.DataFrame({
                'Negative':{'True':FalseNegative,'False':TrueNegative},
                'Positive':{'True':TruePositive,'False':FalsePositive}                            
                })        
    print(_confusion_matrix)
 
#画ROC曲线图
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt 

fpr0,tpr0,thresh0=roc_curve(y,y_pred_old)
fpr,tpr,thresh=roc_curve(y,y_pred)#False Positive Rate与True Positive Rate
Auc=auc(fpr,tpr)
Auc0=auc(fpr0,tpr0)
lw=2
plt.figure(figsize=[10, 10])
plt.plot(fpr0,tpr0,color='purple',lw=lw,label='ROC curve before (area = %.2f)'%Auc0) #剔除变量之前
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve after (area = %.2f)'%Auc) #剔除变量之后
plt.plot([0, 1],[0, 1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC curve')

print(logit_new.summary())    
plt.show()