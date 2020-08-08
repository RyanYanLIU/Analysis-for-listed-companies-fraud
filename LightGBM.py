import pandas as pd 
data=pd.read_csv('data_revised.csv',index_col=0)
data.fillna(0,inplace=True)

from sklearn.metrics import mean_squared_error,accuracy_score,roc_curve,auc
import matplotlib.pyplot as plt 
features=data.drop(['label','year'],axis=1)
target=data.label
def func(fea,tar):
    import lightgbm as lgb
    #Define the training set and testing set 
    n=int(len(fea)*0.8)
    X_train=fea[:n]
    X_test=fea[n:]
    y_train=tar[:n]
    y_test=tar[n:]

    fea_name=list(data.columns[:-2])
    train_data=lgb.Dataset(X_train,y_train,feature_name=fea_name)
    train_dev=lgb.Dataset(X_test,y_test,reference=train_data)
    params = {
        'task':'train',
        'boosting_type':'gbdt',
        'metric': {'l2','fair'},
        'num_leaves':20,
        'num_threads':8,
        'learning_rate':0.5,
        'feature_fraction':0.3,
        'bagging_fraction':0.8
        }
    gbm=lgb.train(params,train_data,num_boost_round=100,valid_sets=train_dev,early_stopping_rounds=10)
    y_pred=gbm.predict(X_test,num_iteration=gbm.best_iteration)
    #Plot ROC Curve
    fpr,tpr,threshold=roc_curve(y_test,y_pred)
    lw=2
    AUC=auc(fpr,tpr)
    plt.figure(figsize=(20,20))
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve area = %0.2f'%AUC) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1],[0, 1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    gbm.save_model('model.txt')
func(features,target)
