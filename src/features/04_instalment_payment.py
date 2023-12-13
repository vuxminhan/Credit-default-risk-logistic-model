import pandas as pd
import pickle
def installments_payments(ins):

    # Read the installments_payments.csv

    ins['NEW_DAYS_PAID_EARLIER'] = ins['DAYS_INSTALMENT']-ins['DAYS_ENTRY_PAYMENT']

    # Whether each installment payment is late or not is 1: paid late, 0: represents early payment.
    ins['NEW_NUM_PAID_LATER'] = ins['NEW_DAYS_PAID_EARLIER'].map(lambda x: 1 if x<0 else 0)

    # Agrregation
    agg_list = {'NUM_INSTALMENT_VERSION':['nunique'],
               'NUM_INSTALMENT_NUMBER':'max',
               'DAYS_INSTALMENT':['min','max'],
               'DAYS_ENTRY_PAYMENT':['min','max'],
               'AMT_INSTALMENT':['min','max','sum','mean'],
               'AMT_PAYMENT':['min','max','sum','mean'],
               'NEW_DAYS_PAID_EARLIER':'mean',
               'NEW_NUM_PAID_LATER':'sum'}


    ins_agg = ins.groupby('SK_ID_PREV').agg(agg_list)


    # Multi index problemi cözümü
    ins_agg.columns = pd.Index(["INS_" + e[0] + '_' + e[1].upper() for e in ins_agg.columns.tolist()])

    # drop variables 
    ins_agg.drop(['INS_DAYS_INSTALMENT_MIN',
                   'INS_DAYS_INSTALMENT_MAX',
                   'INS_DAYS_ENTRY_PAYMENT_MIN',
                   'INS_DAYS_ENTRY_PAYMENT_MAX'],axis=1,inplace=True)

    # Loan payment percentage and total remaining debt
    ins_agg['INS_NEW_PAYMENT_PERC'] = ins_agg['INS_AMT_PAYMENT_SUM'] / ins_agg['INS_AMT_INSTALMENT_SUM']
    ins_agg['INS_NEW_PAYMENT_DIFF'] = ins_agg['INS_AMT_INSTALMENT_SUM'] - ins_agg['INS_AMT_PAYMENT_SUM']
    
    agg_list_previous_application = {}
    
    for col in ins_agg.columns:
        agg_list_previous_application[col] = ['mean',"min","max","sum"]
    
    ins_agg.reset_index(inplace = True) 
    
    return agg_list_previous_application, ins_agg


if __name__ == "__main__":
    ins = pd.read_csv('data/raw/dseb63_installments_payments.csv')
    agg_list_previous_application, ins_agg = installments_payments(ins)
    with open('data/interim/agg_list_previous_application.pkl', 'wb') as f:
        pickle.dump(agg_list_previous_application, f)