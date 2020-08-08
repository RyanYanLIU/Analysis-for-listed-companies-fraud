import pandas as pd 
import numpy as np 
#Compare the dropping rating for each company on each time slot
data0=pd.read_csv('ratings.csv',index_col=None)
#Create 2 new lists for rating level of differenet companies
high_rating=['AAA','Aa1','Aa2','AAA-','AA+','AA']
data0=data0.dropna(how='any')

data0['Cur_score']=data0['B_INFO_CREDITRATING'].apply(lambda x:1 if x in high_rating else 0)
data0['Pre_score']=data0['B_INFO_PRECREDITRATING'].apply(lambda x:1 if x in high_rating else 0)
data0['label']=(data0['Cur_score']-data0['Pre_score']).apply(lambda x:1 if x<0 else 0)

low_grade=data0.loc[data0['label']==1]#之前评级高，现在评级已经下降的公司
def func(company,date):
    df=low_grade.loc[low_grade['S_INFO_COMPCODE']==company]
    if len(df)>0: #df表格存在的情况下
        df=df.loc[df['ANN_DT'].apply(lambda x:True if date+1>=x//10000>date else False)] #寻找什么时间发生评级下调（利用年月日时间将rate和features两表格连接）
        if len(df)>0: #这个数值存在
            return 1
        else:
            return 0
    else:
        return 0

def func1(company,date): # data 里面选出[company]在评级下降日期[之前][最近的]评级是否为good，好公司返回true
    # data : 2012
    # ANN_DT: 20131201
    y=data0.loc[data0['S_INFO_COMPCODE']==company]
    if len(y)>0:
        y=y.loc[y['ANN_DT'].apply(lambda a: True if a//10000<=date else False)]
        if len(y)>0:
            y=y.sort_values(by='ANN_DT')
            return y.iloc[-1]['B_INFO_CREDITRATING'] in high_rating
        else:
            return False
    else:
        return False

features=pd.DataFrame()
for z in [2011,2012,2013,2014,2015,2016]:
    data=pd.read_csv('report_'+str(z)+'1231.csv',index_col=0) 
    #Create another dataframe in order to do division later
    j=z+1
    data1=pd.read_csv('report_'+str(j)+'1231.csv',index_col=0)  
    #dataframe for data
    #第t年的应收账款
    dsr_t=(data['ACCT_RCV']+data['DVD_RCV']+data['INT_RCV']+data['NOTES_RCV']+data['OTH_RCV'])/data['TOT_OPER_REV']
    #第t年毛利率
    gmi_t=data['TOT_PROFIT']/data['TOT_OPER_REV']
    #第t年无形资产占比
    aqi_t=(1-(data['FIX_ASSETS']+data['TOT_CUR_ASSETS']))/data['TOT_ASSETS']
    #第t年销售增长率（其余年限的未知，因为不知道此表格的公司含义）
    sgi_t=data['TOT_OPER_REV']
    #第t年销售管理
    sgai_t=(data['LESS_GERL_ADMIN_EXP']+data['LESS_SELLING_DIST_EXP'])/data['TOT_OPER_REV']
    #第t年杠杆（负债/总资产）
    lgvi_t=(data['TOT_LIAB']+data['LT_BORROW'])/data['TOT_ASSETS']
    #TATA(第t年预提)
    tata_t=(data['OPER_PROFIT']-data['NET_CASH_FLOWS_OPER_ACT'])/data['TOT_ASSETS']
    #DEPI(折旧)
    depi_t=data['DEPR_FA_COGA_DPBA']/(data['DEPR_FA_COGA_DPBA']+data['FIX_ASSETS'])

    #Create a new 8-variable dataframe of data
    df=pd.DataFrame([gmi_t,aqi_t,sgi_t,sgai_t,lgvi_t,tata_t,depi_t],
                    index=['GMI','AQI','SGI','SGAI','LGVI','TATA','DEPI']).T
  
    #Create another dataframe of data1
    #第t年的应收账款
    dsr_t1=(data1['ACCT_RCV']+data1['DVD_RCV']+data1['INT_RCV']+data1['NOTES_RCV']+data1['OTH_RCV'])/data1['TOT_OPER_REV']
    #第t年毛利率
    gmi_t1=data1['TOT_PROFIT']/data1['TOT_OPER_REV']
    #第t年无形资产占比
    aqi_t1=(1-(data1['FIX_ASSETS']+data1['TOT_CUR_ASSETS']))/data1['TOT_ASSETS']
    #第t年销售增长率（其余年限的未知，因为不知道此表格的公司含义）
    sgi_t1=data1['TOT_OPER_REV']
    #第t年销售管理
    sgai_t1=(data1['LESS_GERL_ADMIN_EXP']+data1['LESS_SELLING_DIST_EXP'])/data1['TOT_OPER_REV']
    #第t年杠杆（负债/总资产）
    lgvi_t1=(data1['TOT_LIAB']+data1['LT_BORROW'])/data1['TOT_ASSETS']
    #TATA(第t年预提)
    tata_t1=(data1['OPER_PROFIT']-data1['NET_CASH_FLOWS_OPER_ACT'])/data1['TOT_ASSETS']
    #DEPI(折旧)
    depi_t1=data1['DEPR_FA_COGA_DPBA']/(data1['DEPR_FA_COGA_DPBA']+data1['FIX_ASSETS'])

    #流动资产变化率（(流动总资产-流动总负债)）/总资产
    Ratio_of_Cur=data1['TOT_CUR_ASSETS']-data1['TOT_CUR_LIAB']/data1['TOT_ASSETS']
    #软资产比例
    Ratio_of_SoftAssets=(data1['TOT_ASSETS']-data1['FIX_ASSETS']-data1['CONST_IN_PROG']-
                        data1['PROJ_MATL']-data1['MONETARY_CAP'])/data1['TOT_ASSETS']
    #负债率（总利润+折旧）/总负债
    Ratio_of_Lev=(data1['OPER_PROFIT']+data1['DEPR_FA_COGA_DPBA'])/data1['TOT_CUR_LIAB']
    #资产率(总利润+利息+折旧)/总资产
    Ratio_of_Ast=(data1['OPER_PROFIT']+data1['DEPR_FA_COGA_DPBA']+data1['INT_RCV'])/data1['TOT_ASSETS']
    #总利润的正负值关系
    Total_Profit_binary=data1['TOT_PROFIT'].apply(lambda x: 1 if x>=0 else 0)
    #资产大小
    Asset_size=data1['TOT_ASSETS'].apply(np.log)
    #业务毛收入
    Rev=data1['TOT_OPER_REV'].apply(np.log)
    #Create a new 8-variable dataframe of data
    df1=pd.DataFrame([gmi_t1,aqi_t1,sgi_t1,sgai_t1,lgvi_t1,tata_t1,depi_t1],
                    index=['GMI','AQI','SGI','SGAI','LGVI','TATA','DEPI']).T
    #To do the division
    aa=df1.div(df,axis=0)
    aa.dropna(how='all',axis=1)
    aa['ROL']=Ratio_of_Lev
    aa['ROC']=Ratio_of_Cur
    aa['ROS']=Ratio_of_SoftAssets
    aa['ROA']=Ratio_of_Ast
    aa['Revenue']=Rev
    aa['TOT_PROFIT_label']=Total_Profit_binary
    aa['TOT_SIZE_ASSETS']=Asset_size
    aa['year']=j
    features=pd.concat([features,aa],axis=0)#Dataframe进行拼接
    
features.to_csv('before.csv')
print('Step 1')
features=features[features.apply(lambda z:func1(z.name,z['year']),axis=1)]
features.to_csv('after.csv')

print('Step 2')
features['label']=features.apply(lambda z:func(z.name,z['year']),axis=1)

features.to_csv('data_1.csv')