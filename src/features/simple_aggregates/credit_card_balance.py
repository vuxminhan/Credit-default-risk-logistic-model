import pandas as pd


def credit_card_balance(CCB):


    CCB = pd.get_dummies(CCB, columns= ['NAME_CONTRACT_STATUS'] )  # artik tumu sayisal 

    dropthis = ['NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Demand',
           'NAME_CONTRACT_STATUS_Refused', 'NAME_CONTRACT_STATUS_Sent proposal',
           'NAME_CONTRACT_STATUS_Signed' ]

    CCB = CCB.drop(dropthis, axis=1)
    
    # Fill in the median value for the numerical values
    for col in ['AMT_INST_MIN_REGULARITY', 'CNT_INSTALMENT_MATURE_CUM', 'AMT_PAYMENT_CURRENT']:
        CCB[col].fillna(CCB[col].mean(), inplace=True)

    # Fill in the value 0 for columns related to transactions
    for col in ['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 
            'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT']:
        CCB[col].fillna(0, inplace=True)
    
        # Creating new columns
    CCB['CREDIT_UTILIZATION'] = CCB['AMT_BALANCE'] - CCB['AMT_CREDIT_LIMIT_ACTUAL']
    CCB['MIN_PAYMENT_VS_DRAWINGS'] = CCB['AMT_INST_MIN_REGULARITY'] - CCB['AMT_DRAWINGS_CURRENT']
    CCB['PAYMENT_VS_DRAWINGS'] = CCB['AMT_PAYMENT_TOTAL_CURRENT'] - CCB['AMT_DRAWINGS_CURRENT']
    CCB['PAYMENT_VS_TOTAL_RECEIVABLE'] = CCB['AMT_PAYMENT_TOTAL_CURRENT'] - CCB['AMT_TOTAL_RECEIVABLE']
    CCB['PAYMENT_VS_BALANCE'] = CCB['AMT_PAYMENT_TOTAL_CURRENT'] - CCB['AMT_BALANCE']
    CCB['PAYMENT_VS_MIN_INSTALLMENT'] = CCB['AMT_PAYMENT_TOTAL_CURRENT'] - CCB['AMT_INST_MIN_REGULARITY']
    CCB['OVERDRAFT_AMOUNT'] = CCB['AMT_DRAWINGS_CURRENT'] - CCB['AMT_CREDIT_LIMIT_ACTUAL']
    CCB['BALANCE_VS_TOTAL_RECEIVABLE'] = CCB['AMT_BALANCE'] - CCB['AMT_TOTAL_RECEIVABLE']
    CCB['SUM_ALL_AMT_DRAWINGS'] = CCB[['AMT_DRAWINGS_ATM_CURRENT', 
                                                   'AMT_DRAWINGS_CURRENT', 
                                                   'AMT_DRAWINGS_OTHER_CURRENT', 
                                                   'AMT_DRAWINGS_POS_CURRENT']].sum(axis=1)
    
    CCB['SUM_ALL_CNT_DRAWINGS'] = CCB[['CNT_DRAWINGS_ATM_CURRENT', 
                                                   'CNT_DRAWINGS_CURRENT', 
                                                   'CNT_DRAWINGS_OTHER_CURRENT', 
                                                   'CNT_DRAWINGS_POS_CURRENT']].sum(axis=1)
    
    CCB['RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS'] = CCB['SUM_ALL_AMT_DRAWINGS'] / CCB['SUM_ALL_CNT_DRAWINGS']
    
    grp = CCB.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].agg('nunique').reset_index().rename(index=str, columns={'SK_ID_PREV': 'NUMBER_OF_LOANS'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    
    grp = CCB.groupby(by = ['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index = str, columns = {'SK_ID_PREV': 'NUMBER_OF_LOANS_PER_CUSTOMER'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')

    grp = CCB.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index = str, columns = {'CNT_INSTALMENT_MATURE_CUM': 'NUMBER_OF_INSTALMENTS'})
    grp1 = grp.groupby(by = ['SK_ID_CURR'])['NUMBER_OF_INSTALMENTS'].sum().reset_index().rename(index = str, columns = {'NUMBER_OF_INSTALMENTS': 'TOTAL_INSTALMENTS'})
    CCB = CCB.merge(grp1, on = ['SK_ID_CURR'], how = 'left')

    CCB['INSTALLMENTS_PER_LOAN'] = (CCB['TOTAL_INSTALMENTS']/CCB['NUMBER_OF_LOANS_PER_CUSTOMER']).astype('uint32')


    # This function calculates how many times payments have been delayed # Function to calculate number of times Days Past Due occurred
#     def geciken_gun_hesapla(DPD):

#         # DPD is a series of values of SK_DPD for each of the groupby combination 
#         # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO
#         x = DPD.tolist()
#         c = 0
#         for i,j in enumerate(x):
#             if j != 0:
#                 c += 1  
#         return c 

#     grp = CCB.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: geciken_gun_hesapla(x.SK_DPD)).reset_index().rename(index = str, columns = {0: 'NUMBER_OF_DPD'})
#     grp1 = grp.groupby(by = ['SK_ID_CURR'])['NUMBER_OF_DPD'].mean().reset_index().rename(index = str, columns = {'NUMBER_OF_DPD' : 'DPD_COUNT'})
    CCB['DPD'] = (CCB['SK_DPD'] > 0).astype(int)
    grp = CCB.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['DPD'].sum().reset_index().rename(columns={'DPD': 'NUMBER_OF_DPD'})
    grp1 = grp.groupby('SK_ID_CURR')['NUMBER_OF_DPD'].mean().reset_index().rename(columns={'NUMBER_OF_DPD': 'DPD_COUNT'})

    
    CCB = CCB.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
    
    
    
    def f(min_pay, total_pay):

        M = min_pay.tolist()
        T = total_pay.tolist()
        P = len(M)        # P: taksit sayisi
        c = 0 
        # Find the count of transactions when Payment made is less than Minimum Payment 
        for i in range(len(M)):
            if T[i] < M[i]:
                c += 1  
        return (100*c)/P

    grp = CCB.groupby(by = ['SK_ID_CURR']).apply(lambda x: f(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index = str, columns = { 0 : 'PERCENTAGE_MIN_MISSED_PAYMENTS'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')

    grp = CCB.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_ATM_CURRENT' : 'DRAWINGS_ATM'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')


    grp = CCB.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'DRAWINGS_TOTAL'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')


    CCB['CASH_CARD_RATIO1'] = (CCB['DRAWINGS_ATM']/CCB['DRAWINGS_TOTAL'])*100  # ATM den cektigi nakit / toplam cektigi
    del CCB['DRAWINGS_ATM']
    del CCB['DRAWINGS_TOTAL']

    grp = CCB.groupby(by = ['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'CASH_CARD_RATIO1' : 'CASH_CARD_RATIO'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')


    grp = CCB.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'TOTAL_DRAWINGS'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')


    grp = CCB.groupby(by = ['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'CNT_DRAWINGS_CURRENT' : 'NUMBER_OF_DRAWINGS'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    

    CCB['DRAWINGS_RATIO1'] = (CCB['TOTAL_DRAWINGS']/CCB['NUMBER_OF_DRAWINGS'])*100     # yuzdelik degil, genisleme yapmis
    del CCB['TOTAL_DRAWINGS']
    del CCB['NUMBER_OF_DRAWINGS']


    grp = CCB.groupby(by = ['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'DRAWINGS_RATIO1' : 'DRAWINGS_RATIO'})
    CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')

    del CCB['DRAWINGS_RATIO1']

    CCB['CC_COUNT'] = CCB.groupby('SK_ID_CURR').size()
    
    # This new feature shows the level of credit limit usage compared to the current balance on the credit card
#     CCB['CREDIT_CARD_BALANCE_RATIO'] = CCB.groupby(
#     by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()
    grouped = CCB.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL'])[['AMT_BALANCE']].max().reset_index()
    grouped['CREDIT_CARD_BALANCE_RATIO'] = grouped['AMT_BALANCE'] / grouped['AMT_CREDIT_LIMIT_ACTUAL']
    CCB = CCB.merge(grouped[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL', 'CREDIT_CARD_BALANCE_RATIO']], on=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL'], how='left')

    CCB['INSTALMENTS_PER_LOAN'] = CCB['TOTAL_INSTALMENTS'] / CCB['NUMBER_OF_LOANS']
     
    CCB_agg = CCB.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE':["sum","mean"], 
        'AMT_BALANCE':["sum","mean","min","max", "std"],
        'AMT_CREDIT_LIMIT_ACTUAL':["sum","mean", "max", "min"], 

        'AMT_DRAWINGS_ATM_CURRENT':["sum","mean","min","max"],
        'AMT_DRAWINGS_CURRENT':["sum","mean","min","max"], 
        'AMT_DRAWINGS_OTHER_CURRENT':["sum","mean","min","max"],
        'AMT_DRAWINGS_POS_CURRENT':["sum","mean","min","max"], 
        'AMT_INST_MIN_REGULARITY':["sum","mean","min","max"],
        'AMT_PAYMENT_CURRENT':["sum","mean","min","max"], 
        'AMT_PAYMENT_TOTAL_CURRENT':["sum","mean","min","max"],
        'AMT_RECEIVABLE_PRINCIPAL':["sum","mean","min","max"], 
        'AMT_RECIVABLE':["sum","mean","min","max"], 
        'AMT_TOTAL_RECEIVABLE':["sum","mean","min","max"],

        'CNT_DRAWINGS_ATM_CURRENT':["sum","mean"], 
        'CNT_DRAWINGS_CURRENT':["sum","mean","max"],
        'CNT_DRAWINGS_OTHER_CURRENT':["mean","max"], 
        'CNT_DRAWINGS_POS_CURRENT':["sum","mean","max"],
        'CNT_INSTALMENT_MATURE_CUM':["sum","mean","max","min"],    
        'SK_DPD':["sum","mean","max"], 
        'SK_DPD_DEF':["sum","mean","max"],

        'NAME_CONTRACT_STATUS_Active':["sum","mean","min","max"], 
        'INSTALLMENTS_PER_LOAN':["sum","mean","min","max"],
        'NUMBER_OF_LOANS_PER_CUSTOMER':["mean"], 
        'DPD_COUNT':["mean"],
        'PERCENTAGE_MIN_MISSED_PAYMENTS':["mean"], 
        'CASH_CARD_RATIO':["mean"], 
        'DRAWINGS_RATIO':["mean"],
        'CREDIT_CARD_BALANCE_RATIO': ["mean", "max", "min"],
     'RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS': ['min', 'max', 'mean']})


    CCB_agg.columns = pd.Index(['CCB_' + e[0] + "_" + e[1].upper() for e in CCB_agg.columns.tolist()])

    CCB_agg.reset_index(inplace = True)
    
    return CCB_agg

if __name__ == "__main__":
    CCB = pd.read_csv('data/raw/dseb63_credit_card_balance.csv')
    CCB_agg = credit_card_balance(CCB)
    CCB_agg.to_csv('data/interim/CCB_agg.csv', index=False)