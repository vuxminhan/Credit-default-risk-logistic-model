import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def previous_application(df_prev, agg_list_previous_application):



    # Separation of "WEEKDAY_APPR_PROCESS_START" variable into two categories as WEEK_DAY and WEEKEND

    df_prev["WEEKDAY_APPR_PROCESS_START"] = df_prev["WEEKDAY_APPR_PROCESS_START"].replace(['MONDAY','TUESDAY', 'WEDNESDAY','THURSDAY','FRIDAY'], 'WEEK_DAY')
    df_prev["WEEKDAY_APPR_PROCESS_START"] = df_prev["WEEKDAY_APPR_PROCESS_START"].replace(['SATURDAY', 'SUNDAY'], 'WEEKEND')

    # Separation of the "HOUR_APPR_PROCESS_START" variable into two categories: working_hours and off_hours
    a = [8,9,10,11,12,13,14,15,16,17]
    df_prev["HOUR_APPR_PROCESS_START"] = df_prev["HOUR_APPR_PROCESS_START"].replace(a, 'working_hours')

    b = [18,19,20,21,22,23,0,1,2,3,4,5,6,7]
    df_prev["HOUR_APPR_PROCESS_START"] = df_prev["HOUR_APPR_PROCESS_START"].replace(b, 'off_hours')


    # DAYS_DECISION values less than 1 year were given the value 1, and those greater than 1 year were given the value 0.
    df_prev["DAYS_DECISION"] = [1 if abs(i/(12*30)) <=1 else 0 for i in df_prev.DAYS_DECISION]

    # Separating the "NAME_TYPE_SUITE" variable into two categories as alone and not_alone

    df_prev["NAME_TYPE_SUITE"] = df_prev["NAME_TYPE_SUITE"].replace('Unaccompanied', 'alone')

    b = ['Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']
    df_prev["NAME_TYPE_SUITE"] = df_prev["NAME_TYPE_SUITE"].replace(b, 'not_alone')



    # These values in the "NAME_GOODS_CATEGORY" variable will be categorized as others
    a = ['Auto Accessories', 'Jewelry', 'Homewares', 'Medical Supplies', 'Vehicles', 'Sport and Leisure', 
         'Gardening', 'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 'Fitness', 'Additional Service', 
         'Education', 'Weapon', 'Insurance', 'House Construction', 'Animals'] 
    df_prev["NAME_GOODS_CATEGORY"] = df_prev["NAME_GOODS_CATEGORY"].replace(a, 'others')

    # "These values in the "NAME_SELLER_INDUSTRY" variable will be categorized as others
    a = ['Auto technology', 'Jewelry', 'MLM partners', 'Tourism'] 
    df_prev["NAME_SELLER_INDUSTRY"] = df_prev["NAME_SELLER_INDUSTRY"].replace(a, 'others')
    # Derives the variable containing the ratio of the loan requested to the loan granted
    df_prev["LOAN_RATE"] = df_prev.AMT_APPLICATION/df_prev.AMT_CREDIT

    # NEW VARIABLES

    # Derives the variable containing the ratio of the loan requested to the loan granted
    df_prev["NEW_LOAN_RATE"] = df_prev.AMT_APPLICATION/df_prev.AMT_CREDIT

    # The churn_prev variable is derived to indicate whether the payment date is late.
    # 1= delayed, 0 = not delayed, NaN = null
    k = df_prev.DAYS_LAST_DUE_1ST_VERSION - df_prev.DAYS_LAST_DUE
    df_prev["NEW_CHURN_PREV"] = [1 if i >= 0 else (0 if i < 0  else "NaN") for i in k]


    # NEW_INSURANCE variable has been defined to be used instead of NFLAG_INSURED_ON_APPROVAL variable.
    df_prev[(df_prev['AMT_CREDIT'] == 0) | (df_prev['AMT_GOODS_PRICE'] == 0)]['NEW_INSURANCE'] = np.nan
    df_prev['sigorta_miktari'] = df_prev['AMT_CREDIT'] - df_prev['AMT_GOODS_PRICE']
    df_prev["NEW_INSURANCE"] = df_prev['sigorta_miktari'].apply(lambda x: 1 if x > 0 else (0 if x <= 0 else np.nan))
    df_prev.drop('sigorta_miktari', axis=1, inplace=True)



# ADJUSTMENT_DIRECTION varible indicates whether the final credit amount is equal to clients' application  #PREV APPLICATION
    df_prev['AMT_DIFF_PERCENT'] = ((df_prev['AMT_CREDIT'] - df_prev['AMT_APPLICATION']) / df_prev['AMT_APPLICATION']) * 100
    df_prev['ADJUSTMENT_DIRECTION'] = (df_prev['AMT_CREDIT'] - df_prev['AMT_APPLICATION']) >= 0
    df_prev['ADJUSTMENT_DIRECTION'] = df_prev['ADJUSTMENT_DIRECTION'].map({'True': 1, 'False': 0})
#New features
    df_prev['PREVIOUS_TERM'] = df_prev['AMT_CREDIT'] / df_prev['AMT_ANNUITY']
    df_prev['PREVIOUS_AMT_TO_APPLICATION'] = df_prev['AMT_CREDIT'] / df_prev['AMT_APPLICATION']
    df_prev['PREVIOUS_CREDIT_TO_PRICE'] = df_prev['AMT_GOODS_PRICE'] / df_prev['AMT_CREDIT']
    df_prev['PREVIOUS_DAYSLASTDUE1ST_DAYSFIRSTDUE_DIFF'] = df_prev['DAYS_LAST_DUE_1ST_VERSION'] - df_prev['DAYS_FIRST_DUE']
    df_prev['PREVIOUS_DAYSLASTDUE_DAYSFIRSTDUE_DIFF'] = df_prev['DAYS_LAST_DUE'] - df_prev['DAYS_FIRST_DUE']
    df_prev['PREVIOUS_DAYSLASTDUE_DAYSLASTDUE1ST_DIFF'] = df_prev['DAYS_LAST_DUE'] - df_prev['DAYS_LAST_DUE_1ST_VERSION']
    
    
    #Interest rate
    df_prev['INTEREST'] = df_prev['CNT_PAYMENT']*df_prev['AMT_ANNUITY'] - df_prev['AMT_CREDIT']
    df_prev['INTEREST_RATE'] = 2*12*df_prev['INTEREST']/(df_prev['AMT_CREDIT']*(df_prev['CNT_PAYMENT']+1))
    df_prev['INTEREST_SHARE'] = df_prev['INTEREST']/df_prev['AMT_CREDIT']
    
    drop_list = ['AMT_DOWN_PAYMENT', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'PRODUCT_COMBINATION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL']
    df_prev.drop(drop_list, axis = 1, inplace = True)

    # df_prev[df_prev['INTEREST_RATE']==-1]=np.nan It holds the names of categorical variables in the Previous table.
    category_columns=[]
    for i in df_prev.columns:
        if df_prev[i].dtypes == "O":
            category_columns.append(i)

    df_prev = pd.get_dummies(df_prev, columns = category_columns )

    prev_agg_list = {"SK_ID_CURR":["count"], 
                "AMT_ANNUITY":["max"],
                "AMT_APPLICATION":["min","mean","max"],
                "AMT_CREDIT":["max"], 
                "AMT_GOODS_PRICE":["sum", "mean"],
                "NFLAG_LAST_APPL_IN_DAY":["sum","mean"], 
                "RATE_DOWN_PAYMENT":["sum", "mean"],
                'AMT_DIFF_PERCENT':['max','min'],
                'ADJUSTMENT_DIRECTION':['max','min'],
                "RATE_INTEREST_PRIMARY":["sum", "mean"],
                "RATE_INTEREST_PRIVILEGED":["sum", "mean"],
                "DAYS_DECISION":["sum"],
                "NEW_LOAN_RATE":["sum", "mean", "min", "max"],
                "NEW_INSURANCE":["sum", "mean"],
                #"INTEREST_RATE":["sum", "mean", "min", "max"],
                "NAME_CONTRACT_TYPE_Cash loans":["sum", "mean"],
                "NAME_CONTRACT_TYPE_Consumer loans":["sum", "mean"],
                "NAME_CONTRACT_TYPE_Revolving loans":["sum", "mean"],
                "NAME_CONTRACT_TYPE_XNA":["sum", "mean"],
                "WEEKDAY_APPR_PROCESS_START_WEEKEND":["sum", "mean"],
                "WEEKDAY_APPR_PROCESS_START_WEEK_DAY":["sum", "mean"],
                "HOUR_APPR_PROCESS_START_off_hours":["sum", "mean"],
                "HOUR_APPR_PROCESS_START_working_hours":["sum", "mean"],
                "FLAG_LAST_APPL_PER_CONTRACT_N":["sum", "mean"],
                "FLAG_LAST_APPL_PER_CONTRACT_Y":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Building a house or an annex":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Business development":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a garage":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a home":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a new car":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Buying a used car":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Car repairs":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Education":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Everyday expenses":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Furniture":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Gasification / water supply":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Hobby":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Journey":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Medicine":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Money for a third person":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Other":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Payments on other loans":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Refusal to name the goal":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Repairs":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Urgent needs":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_XAP":["sum", "mean"],
                "NAME_CASH_LOAN_PURPOSE_XNA":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Approved":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Canceled":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Refused":["sum", "mean"],
                "NAME_CONTRACT_STATUS_Unused offer":["sum", "mean"],
                "NAME_PAYMENT_TYPE_Cash through the bank":["sum", "mean"],
                "NAME_PAYMENT_TYPE_Cashless from the account of the employer":["sum", "mean"],
                "NAME_PAYMENT_TYPE_Non-cash from your account":["sum", "mean"],
                "NAME_PAYMENT_TYPE_XNA":["sum", "mean"],
                "CODE_REJECT_REASON_CLIENT":["sum", "mean"],
                "CODE_REJECT_REASON_HC":["sum", "mean"],
                "CODE_REJECT_REASON_LIMIT":["sum", "mean"],
                "CODE_REJECT_REASON_SCO":["sum", "mean"],
                "CODE_REJECT_REASON_SCOFR":["sum", "mean"],
                "CODE_REJECT_REASON_SYSTEM":["sum", "mean"],
                "CODE_REJECT_REASON_VERIF":["sum", "mean"],
                "CODE_REJECT_REASON_XAP":["sum", "mean"],
                "CODE_REJECT_REASON_XNA":["sum", "mean"],
                "NAME_TYPE_SUITE_alone":["sum", "mean"],
                "NAME_TYPE_SUITE_not_alone":["sum", "mean"],
                "NAME_CLIENT_TYPE_New":["sum", "mean"],
                "NAME_CLIENT_TYPE_Refreshed":["sum", "mean"],
                "NAME_CLIENT_TYPE_Repeater":["sum", "mean"],
                "NAME_CLIENT_TYPE_XNA":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Audio/Video":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Clothing and Accessories":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Computers":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Construction Materials":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Consumer Electronics":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Furniture":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Mobile":["sum", "mean"],
                "NAME_GOODS_CATEGORY_Photo / Cinema Equipment":["sum", "mean"],
                "NAME_GOODS_CATEGORY_XNA":["sum", "mean"],
                "NAME_GOODS_CATEGORY_others":["sum", "mean"],
                "NAME_PORTFOLIO_Cards":["sum", "mean"],
                "NAME_PORTFOLIO_Cars":["sum", "mean"],
                "NAME_PORTFOLIO_Cash":["sum", "mean"],
                "NAME_PORTFOLIO_POS":["sum", "mean"],
                "NAME_PORTFOLIO_XNA":["sum", "mean"],
                "NAME_PRODUCT_TYPE_XNA":["sum", "mean"],
                "NAME_PRODUCT_TYPE_walk-in":["sum", "mean"],
                "NAME_PRODUCT_TYPE_x-sell":["sum", "mean"],
                "CHANNEL_TYPE_AP+ (Cash loan)":["sum", "mean"],
                "CHANNEL_TYPE_Car dealer":["sum", "mean"],
                "CHANNEL_TYPE_Channel of corporate sales":["sum", "mean"],
                "CHANNEL_TYPE_Contact center":["sum", "mean"],
                "CHANNEL_TYPE_Country-wide":["sum", "mean"],
                "CHANNEL_TYPE_Credit and cash offices":["sum", "mean"],
                "CHANNEL_TYPE_Regional / Local":["sum", "mean"],
                "CHANNEL_TYPE_Stone":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Clothing":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Connectivity":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Construction":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Consumer electronics":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Furniture":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_Industry":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_XNA":["sum", "mean"],
                "NAME_SELLER_INDUSTRY_others":["sum", "mean"],
                "NAME_YIELD_GROUP_XNA":["sum", "mean"],
                "NAME_YIELD_GROUP_high":["sum", "mean"],
                "NAME_YIELD_GROUP_low_action":["sum", "mean"],
                "NAME_YIELD_GROUP_low_normal":["sum", "mean"],
                "NAME_YIELD_GROUP_middle":["sum", "mean"],
                "NEW_CHURN_PREV_0":["sum", "mean"],
                "NEW_CHURN_PREV_1":["sum", "mean"],
                "NEW_CHURN_PREV_NaN":["sum", "mean"],
                #New feat
                'PREVIOUS_TERM':['count', 'min','max'],
                'PREVIOUS_AMT_TO_APPLICATION':['count', 'min','max'],
                'PREVIOUS_CREDIT_TO_PRICE':['count', 'min','max'],
                'PREVIOUS_DAYSLASTDUE1ST_DAYSFIRSTDUE_DIFF':['count', 'min','max'],
                'PREVIOUS_DAYSLASTDUE_DAYSFIRSTDUE_DIFF' :['count', 'min','max'],
                'PREVIOUS_DAYSLASTDUE_DAYSLASTDUE1ST_DIFF':['count', 'min','max'],
                'INTEREST' :['count', 'min','max'],
                'INTEREST_RATE'   :['count', 'min','max'],  
                'INTEREST_SHARE':['count', 'min','max'] 
}
    prev_agg_list.update(agg_list_previous_application)
    
    
    return prev_agg_list, df_prev


if __name__ == "__main__":
    df_prev = pd.read_csv('data/raw/dseb63_previous_application.csv')
    with open('data/interim/agg_list_previous_application.pkl', 'rb') as f:
        agg_list_previous_application = pickle.load(f)
    prev_agg_list, df_prev = previous_application(df_prev, agg_list_previous_application)
    with open('data/interim/prev_agg_list.pkl', 'wb') as f:
        pickle.dump(prev_agg_list, f)
    df_prev.to_csv('data/interim/df_prev.csv')