import pandas as pd
import pickle
def pos_cash_balance(agg_list_previous_application, pos):

    
    # Converting Our Categorical Variable to Dummy Variable
    pos = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na = True)
#     pos['INSTALLMENTS_DAYS_DIFF'] = pos['DAYS_INSTALMENT'] - pos['DAYS_ENTRY_PAYMENT']
#     pos['INSTALLMENTS_AMT_INSTALLMENT_TO_PAYMENT'] = pos['AMT_INSTALMENT'] / pos['AMT_PAYMENT']




#     pos['INSTALLMENTS_DAYS_365'] = [x if x >= -365 else np.nan for x in pos.DAYS_ENTRY_PAYMENT]
#     pos['INSTALLMENTS_DAYS_DIFF_LASTYR'] = pos['DAYS_INSTALMENT'] - pos['INSTALLMENTS_DAYS_365']
#     pos.drop('INSTALLMENTS_DAYS_365', axis=1, inplace=True)
    # Aggregation Process - Deduplication
    agg_list = {'MONTHS_BALANCE':['min','max'],
                                            'CNT_INSTALMENT':['min','max'],
                                            'CNT_INSTALMENT_FUTURE':['min','max'],
                                            'SK_DPD':['max','mean'],
                                            'SK_DPD_DEF':['max','mean'],
                                            'NAME_CONTRACT_STATUS_Active':'sum',
                                            'NAME_CONTRACT_STATUS_Amortized debt':'sum',
                                            'NAME_CONTRACT_STATUS_Approved':'sum',
                                            'NAME_CONTRACT_STATUS_Canceled':'sum',
                                            'NAME_CONTRACT_STATUS_Completed':'sum',
                                            'NAME_CONTRACT_STATUS_Demand':'sum',
                                            'NAME_CONTRACT_STATUS_Returned to the store':'sum',
                                            'NAME_CONTRACT_STATUS_Signed':'sum',
                                            'NAME_CONTRACT_STATUS_XNA':'sum',
                                            'NAME_CONTRACT_STATUS_nan':'sum'}
#                                             'INSTALLMENTS_DAYS_DIFF_LASTYR': ['sum','min','max'],
#                                             'INSTALLMENTS_DAYS_DIFF': ['sum','min','max'],
#                                             'INSTALLMENTS_AMT_INSTALLMENT_TO_PAYMENT':['sum','min','max']}
    
    
    pos_agg = pos.groupby('SK_ID_PREV').agg(agg_list)

    # Converting multilayer index to one-dimensional index
    pos_agg.columns= pd.Index(["POS_" + e[0] + '_' + e[1].upper() for e in pos_agg.columns.tolist()])

    # SK_DPD is 0 in how many credits (SK_DPD gives 0 status when we get MAX)
    # SK_DPD_DEF (returns SK_DPD_DEF_MAX to zero)
    # NAME_CONTRACT_STATUS_Completed_SUM==0 when CNT_INSTALMENT_FUTURE_MIN==0

    pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME']= (pos_agg['POS_CNT_INSTALMENT_FUTURE_MIN']==0) & (pos_agg['POS_NAME_CONTRACT_STATUS_Completed_SUM']==0)


    # 1: loan not closed on time 0: loan closed on time

    pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME']=pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'].astype(int)

    pos_agg.drop(['POS_NAME_CONTRACT_STATUS_Approved_SUM',
                   'POS_NAME_CONTRACT_STATUS_Amortized debt_SUM',
                   'POS_NAME_CONTRACT_STATUS_Canceled_SUM',
                   'POS_NAME_CONTRACT_STATUS_Returned to the store_SUM',
                   'POS_NAME_CONTRACT_STATUS_Signed_SUM',
                   'POS_NAME_CONTRACT_STATUS_XNA_SUM',
                   'POS_NAME_CONTRACT_STATUS_nan_SUM'],axis=1,inplace=True)

    for col in pos_agg.columns:
        agg_list_previous_application[col] = ['mean',"min","max","sum"]

    pos_agg.reset_index(inplace = True)     
    
    return agg_list_previous_application, pos_agg



if __name__ == "__main__":
    pos = pd.read_csv('data/raw/dseb63_POS_CASH_balance.csv')
    with open('data/interim/agg_list_previous_application.pkl', 'rb') as f:
        agg_list_previous_application = pickle.load(f)
    agg_list_previous_application, pos_agg = pos_cash_balance(agg_list_previous_application,pos)
    print("Pos Cash Balance:", pos_agg.shape)  