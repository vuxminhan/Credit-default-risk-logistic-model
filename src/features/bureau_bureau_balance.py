import pandas as pd

def bureau_bb(bb, bureau):

    bb = pd.get_dummies(bb, dummy_na = True)

    agg_list = {"MONTHS_BALANCE":"count",
                "STATUS_0":["sum","mean"],
                "STATUS_1":["sum"],
                "STATUS_2":["sum"],
                "STATUS_3":["sum"],
                "STATUS_4":["sum"],
                "STATUS_5":["sum"],
                "STATUS_C":["sum","mean"],
                "STATUS_X":["sum","mean"] }

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(agg_list)
    
    
    # Renaming variable names
    bb_agg.columns = pd.Index([col[0] + "_" + col[1].upper() for col in bb_agg.columns.tolist()])

    # Creating a new variable for Status Sum
    bb_agg['NEW_STATUS_SCORE'] = bb_agg['STATUS_1_SUM'] + bb_agg['STATUS_2_SUM']^2 + bb_agg['STATUS_3_SUM']^3 + bb_agg['STATUS_4_SUM']^4 + bb_agg['STATUS_5_SUM']^5

    bb_agg.drop(['STATUS_1_SUM','STATUS_2_SUM','STATUS_3_SUM','STATUS_4_SUM','STATUS_5_SUM'], axis=1,inplace=True)

    bureau['CREDIT_ENDDATE_BINARY'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
    bureau_agg = bureau.merge(bureau.groupby('SK_ID_CURR')['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'}), on='SK_ID_CURR', how='left')
    
#   Handle missing values
#     bureau['AMT_CREDIT_SUM'].fillna(bureau['AMT_CREDIT_SUM'].median(), inplace=True)
#     bureau['DAYS_CREDIT_ENDDATE'].fillna(bureau['DAYS_CREDIT_ENDDATE'].median(), inplace=True)
# #     bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)   # giả sử missing values là các khoản vay đã được thay toán hoàn toàn
#     bureau['AMT_CREDIT_SUM_LIMIT'].fillna(0, inplace=True)  # giả sử cmissing values là những người không có giới hạn tín dụng 
#     bureau['DAYS_ENDDATE_FACT'].fillna(99999, inplace=True) # giả sử missing values là các khoản vay vẫn còn đang hoạt đôngj
#     bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(0, inplace=True) # giả sử missing values là các khoản vay không có lịch sử quá hạn thanh toán
#     bureau['AMT_ANNUITY'].fillna(0, inplace=True)            # giả sử missing values là các khoản vay không có khoản trả góp định kỳ

    bureau_and_bb = bureau_agg.join(bb_agg, how='left', on='SK_ID_BUREAU')
    # BUREAU BALANCE AND BUREAU TABLE
    
    # Reducing the number of classes of the CREDIT TYPE variable to 3
    bureau_and_bb['CREDIT_TYPE'] = bureau_and_bb['CREDIT_TYPE'].replace(['Car loan',
              'Mortgage',
              'Microloan',
              'Loan for business development', 
              'Another type of loan',
              'Unknown type of loan', 
              'Loan for working capital replenishment',
              "Loan for purchase of shares (margin lending)",                                                
              'Cash loan (non-earmarked)', 
              'Real estate loan',
              "Loan for the purchase of equipment", 
              "Interbank credit", 
              "Mobile operator loan"], 'Rare')


    # Reducing the number of classes of the CREDIT_ACTIVE variable to 2 (Would it be more appropriate to include Sold in Closed???)
    bureau_and_bb['CREDIT_ACTIVE'] = bureau_and_bb['CREDIT_ACTIVE'].replace(['Bad debt','Sold'], 'Active')

    # Applying One Hot Encoding to categorical variables in the bureau_bb table
    bureau_and_bb = pd.get_dummies(bureau_and_bb, columns = ["CREDIT_TYPE","CREDIT_ACTIVE"])

    # 99% of the CREDIT CURRENCY variable is currency 1, so we deleted it because we thought it would not be distinctive.
    bureau_and_bb.drop(["SK_ID_BUREAU","CREDIT_CURRENCY"], inplace = True, axis = 1)
    
    
    # New variable showing the average number of months of credit received
    bureau_and_bb["NEW_MONTHS_CREDIT"]= round((bureau_and_bb.DAYS_CREDIT_ENDDATE - bureau_and_bb.DAYS_CREDIT)/30)
    
    # New variable showing the difference between the actual closing date of the credit and the expected closing date of the credit
    bureau_and_bb["CREDIT_AND_DATE_DIFFERENCE"] = round((bureau_and_bb.DAYS_ENDDATE_FACT - bureau_and_bb.DAYS_CREDIT_ENDDATE)/30)
    bureau_and_bb['RATIO_CREDIT_DAY_OVERDUE_TO_90_DAYS'] = bureau_and_bb['CREDIT_DAY_OVERDUE'] / 90
    
    bureau_and_bb['RATIO_AMT_CREDIT_SUM_OVERDUE_TO_CNT_CREDIT_PROLONG'] = bureau_and_bb['AMT_CREDIT_SUM_OVERDUE'] / bureau_and_bb['CNT_CREDIT_PROLONG']
    bureau_and_bb['RATIO_AMT_CREDIT_MAX_OVERDUE_TO_CNT_CREDIT_PROLONG'] = bureau_and_bb['AMT_CREDIT_MAX_OVERDUE'] / bureau_and_bb['CNT_CREDIT_PROLONG']

    agg_list = {
          "SK_ID_CURR":["count"],
          "DAYS_CREDIT":["min","max"],
          "CREDIT_DAY_OVERDUE":["sum","mean","max", "min"],     
          "DAYS_CREDIT_ENDDATE":["max","min"],
          "DAYS_ENDDATE_FACT":["max","min"],
          "AMT_CREDIT_MAX_OVERDUE":["mean","max","min"],
          "CNT_CREDIT_PROLONG":["sum", "mean", "max", "min"],
          "AMT_CREDIT_SUM":["sum","mean","max","min"],            
          "AMT_CREDIT_SUM_DEBT":["sum","mean","max"],
          "AMT_CREDIT_SUM_LIMIT":["sum","mean","max"],
          'AMT_CREDIT_SUM_OVERDUE':["sum","mean","max"], 
          'DAYS_CREDIT_UPDATE':["max","min"],
          'AMT_ANNUITY':["sum","mean"],
          'MONTHS_BALANCE_COUNT':["sum"], 
          'STATUS_0_SUM':["sum"],         
          'STATUS_0_MEAN':["mean"], 
          'STATUS_C_SUM':["sum"], 
          'STATUS_C_MEAN':["mean"],
          'CREDIT_ACTIVE_Active':["sum","mean"], 
          'CREDIT_ACTIVE_Closed':["sum","mean"], 
          'CREDIT_TYPE_Rare':["sum","mean"],      
          'CREDIT_TYPE_Consumer credit':["sum","mean"], 
          'CREDIT_TYPE_Credit card':["sum","mean"],
          'CREDIT_ENDDATE_PERCENTAGE': ["sum","mean","max","min"],
          "NEW_MONTHS_CREDIT":["count","sum","mean","max","min"],
        'RATIO_CREDIT_DAY_OVERDUE_TO_90_DAYS': ["mean","max","min"],
    'RATIO_AMT_CREDIT_SUM_OVERDUE_TO_CNT_CREDIT_PROLONG': ['min', 'max', 'mean'],
        'RATIO_AMT_CREDIT_MAX_OVERDUE_TO_CNT_CREDIT_PROLONG': ['min', 'max', 'mean']}


    # Application of aggregation operations to the bureau _bb_agg table
    bureau_and_bb_agg = bureau_and_bb.groupby("SK_ID_CURR").agg(agg_list).reset_index()


    # Renaming variable names
    bureau_and_bb_agg.columns = pd.Index(["BB_" + col[0] + "_" + col[1].upper() for col in bureau_and_bb_agg.columns.tolist()])

    # New variable showing the difference between the highest and lowest credit a person has received
    bureau_and_bb_agg["BB_NEW_AMT_CREDIT_SUM_RANGE"] = bureau_and_bb_agg["BB_AMT_CREDIT_SUM_MAX"] - bureau_and_bb_agg["BB_AMT_CREDIT_SUM_MIN"]

    # New variable expressing how many months on average people take out a loan
    bureau_and_bb_agg["BB_NEW_DAYS_CREDIT_RANGE"]= round((bureau_and_bb_agg["BB_DAYS_CREDIT_MAX"] - bureau_and_bb_agg["BB_DAYS_CREDIT_MIN"])/(30 * bureau_and_bb_agg["BB_SK_ID_CURR_COUNT"]))
    
    # The ratio of total debt to total credit for each customer
    bureau_and_bb_agg["BB_DEBT_CREDIT_RATIO"] = bureau_and_bb_agg["BB_AMT_CREDIT_SUM_DEBT_SUM"] / bureau_and_bb_agg["BB_AMT_CREDIT_SUM_SUM"]
    
    # The situation of overdue debt for each customer compared to their total debt
    bureau_and_bb_agg["BB_OVERDUE_DEBT_RATIO"] = bureau_and_bb_agg["BB_AMT_CREDIT_SUM_OVERDUE_SUM"] / bureau_and_bb_agg["BB_AMT_CREDIT_SUM_DEBT_SUM"]
    
    # Bureau: Active credits - using only numerical aggregations
    agg_list = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum']
        }

    active = bureau_and_bb[bureau_and_bb['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(agg_list)
    active_agg.columns = pd.Index(['BB_NEW_ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_and_bb_agg.rename(columns = {'BB_SK_ID_CURR_': 'SK_ID_CURR'}, inplace = True)
    bureau_and_bb_agg = bureau_and_bb_agg.join(active_agg, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau_and_bb[bureau_and_bb['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(agg_list)
    closed_agg.columns = pd.Index(['BB_NEW_CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_and_bb_agg = bureau_and_bb_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    return bureau_and_bb_agg


if __name__ == "__main__":
    bureau = pd.read_csv('data/raw/dseb63_bureau.csv')
    bereau_balance = pd.read_csv('data/raw/dseb63_bureau_balance.csv')
    
    processed_df = bureau_bb(bereau_balance, bureau)
    processed_df.to_csv('data/interim/bureau_and_bb_agg.csv', index=False)