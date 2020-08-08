import numpy as np 
import pandas as pd 

#writer=pd.ExcelWriter('Revise.xlsx')
#Ingest the raw data to find the 8-variables
for z in range(1,7):
    data=pd.read_csv('report_201'+str(z)+'1231.csv',index_col=0) 
    #Create another dataframe in order to do division later
    j=z+1
    data1=pd.read_csv('report_201'+str(j)+'1231.csv',index_col=0)  
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
    df=pd.DataFrame([dsr_t,gmi_t,aqi_t,sgi_t,sgai_t,lgvi_t,tata_t,depi_t],
                    index=['DSR','GMI','AQI','SGI','SGAI','LGVI','TATA','DEPI']).T
  
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

    #Create a new 8-variable dataframe of data
    df1=pd.DataFrame([dsr_t1,gmi_t1,aqi_t1,sgi_t1,sgai_t1,lgvi_t1,tata_t1,depi_t1],
                    index=['DSR','GMI','AQI','SGI','SGAI','LGVI','TATA','DEPI']).T
    #To do the division
    aa=df1.div(df,axis=0)
    aa['year']=aa.apply(lambda x:'201'+str(j),axis=1)
    #save into the CSV file
    aa.to_csv('8-variable.csv',mode='a')
'''   
    aa.to_csv(writer,sheet_name=f'{z}')
    writer.save()'''