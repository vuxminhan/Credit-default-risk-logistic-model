
import numpy as np
import pandas as pd
# Add other necessary imports based on the notebook content
# previous_app
previous_application = pd.read_csv('data/raw/dseb63_previous_application.csv')
previous_application = previous_application.sort_values(by = ['SK_ID_CURR','DAYS_FIRST_DUE'])
aggregations_for_previous_application = {'AMT_ANNUITY' : ['mean','sum','max'], 'AMT_GOODS_PRICE' : ['mean','max','sum']}
#grouping the previous applications over SK_ID_CURR while only taking the latest 5 applications
group_last_5 = previous_application.groupby('SK_ID_CURR').tail(5).groupby('SK_ID_CURR').agg(aggregations_for_previous_application)
group_last_5.columns = ['_'.join(ele).upper() + '_LAST_5' for ele in group_last_5.columns]
#grouping the previous applications over SK_ID_CURR while only taking the first 2 applications
group_first_2 = previous_application.groupby('SK_ID_CURR').head(2).groupby('SK_ID_CURR').agg(aggregations_for_previous_application)
group_first_2.columns = ['_'.join(ele).upper() + '_FIRST_2' for ele in group_first_2.columns]
#grouping the previous applications over SK_ID_CURR while taking all the applications into consideration
group_all = previous_application.groupby('SK_ID_CURR').agg(aggregations_for_previous_application)
group_all.columns = ['_'.join(ele).upper() + '_ALL' for ele in group_all.columns]

#merging all the applications
previous_application_aggregated = group_last_5.merge(group_first_2, on = 'SK_ID_CURR', how = 'outer')
previous_application_aggregated = previous_application_aggregated.merge(group_all, on = 'SK_ID_CURR', how = 'outer')# This Python 3 environment comes with many helpful analytics libraries installed


#credit_card

#making the MONTHS_BALANCE Positive
cc_balance = pd.read_csv('data/raw/dseb63_credit_card_balance.csv')
cc_balance['MONTHS_BALANCE'] = np.abs(cc_balance['MONTHS_BALANCE'])
#sorting the DataFrame according to the month of status from oldest to latest, for rolling computations
cc_balance = cc_balance.sort_values(by = ['SK_ID_PREV','MONTHS_BALANCE'], ascending = [1,0])

rolling_columns = [
        'AMT_RECEIVABLE_PRINCIPAL',
        'AMT_RECIVABLE',
        'AMT_TOTAL_RECEIVABLE',
         ]
exp_weighted_columns = ['EXP_' + ele for ele in rolling_columns]
cc_balance[exp_weighted_columns] = cc_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])[rolling_columns].transform(lambda x: x.ewm(alpha = 0.7).mean())


overall_aggregations = {'AMT_RECEIVABLE_PRINCIPAL' : ['sum','mean','max'], 'AMT_RECIVABLE' : ['sum','mean','max'], 'AMT_TOTAL_RECEIVABLE' : ['sum','mean','max']}
cc_balance_aggregated_overall = cc_balance.groupby('SK_ID_PREV').agg(overall_aggregations)
cc_balance_aggregated_overall.columns = ['_'.join(ele).upper() for ele in cc_balance_aggregated_overall.columns]

#-> đoạn này encoding biến categorical sau rồi groupby bằng SK_ID_CURR
cc_aggregated = pd.read_csv("data/interim/binned_numerical_features/processed_credit_card_balance.csv")
cc_aggregated = cc_aggregated.groupby('SK_ID_CURR', as_index = False).mean()


# installment
installments_payments = pd.read_csv('data/raw/dseb63_installments_payments.csv')
installments_payments = installments_payments.sort_values(by = ['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'], ascending = True)

overall_aggregations = {'AMT_PAYMENT' : ['mean', 'sum', 'max'], 'AMT_INSTALMENT' : ['mean', 'sum', 'max']}
limited_period_aggregations = {'AMT_PAYMENT' : ['mean', 'sum', 'max'], 'AMT_INSTALMENT' : ['mean', 'sum', 'max']}


#aggregating installments_payments over SK_ID_PREV for last 1 year installments
group_last_1_year = installments_payments[installments_payments['DAYS_INSTALMENT'] > -365].groupby('SK_ID_PREV').agg(limited_period_aggregations)
group_last_1_year.columns = ['_'.join(ele).upper() + '_LAST_1_YEAR' for ele in group_last_1_year.columns]
#aggregating installments_payments over SK_ID_PREV for first 5 installments
group_first_5_instalments = installments_payments.groupby('SK_ID_PREV', as_index = False).head(5).groupby('SK_ID_PREV').agg(limited_period_aggregations)
group_first_5_instalments.columns = ['_'.join(ele).upper() + '_FIRST_5_INSTALLMENTS' for ele in group_first_5_instalments.columns]
#overall aggregation of installments_payments over SK_ID_PREV
group_overall = installments_payments.groupby(['SK_ID_PREV','SK_ID_CURR'], as_index = False).agg(overall_aggregations)
group_overall.columns = ['_'.join(ele).upper() for ele in group_overall.columns]
group_overall.rename(columns = {'SK_ID_PREV_': 'SK_ID_PREV','SK_ID_CURR_' : 'SK_ID_CURR'}, inplace = True)

#merging all of the above aggregations together
installments_payments_agg_prev = group_overall.merge(group_last_1_year, on = 'SK_ID_PREV', how = 'outer')
installments_payments_agg_prev = installments_payments_agg_prev.merge(group_first_5_instalments, on = 'SK_ID_PREV', how = 'outer')



# bureau
bureau = pd.read_csv('data/raw/dseb63_bureau.csv')
bureau_balance = pd.read_csv('data/raw/dseb63_bureau_balance.csv')
bureau_merged = bureau.merge(bureau_balance, on = 'SK_ID_BUREAU', how = 'outer')
aggregations_CREDIT_ACTIVE = {'AMT_CREDIT_MAX_OVERDUE': ['max','sum'],'AMT_CREDIT_SUM_OVERDUE': ['max','sum'], 'AMT_ANNUITY' : ['mean','sum','max'] }
categories_to_aggregate_on = ['Closed','Active']

bureau_merged_aggregated_credit = pd.DataFrame()
for i, status in enumerate(categories_to_aggregate_on):
    group = bureau_merged[bureau_merged['CREDIT_ACTIVE'] == status].groupby('SK_ID_CURR').agg(aggregations_CREDIT_ACTIVE)
    group.columns = ['_'.join(ele).upper() + '_CREDITACTIVE_' + status.upper() for ele in group.columns]

    if i==0:
        bureau_merged_aggregated_credit = group
    else:
        bureau_merged_aggregated_credit = bureau_merged_aggregated_credit.merge(group, on = 'SK_ID_CURR', how = 'outer')
 #aggregating for remaining categories
bureau_merged_aggregated_credit_rest = bureau_merged[(bureau_merged['CREDIT_ACTIVE'] != 'Active') & 
                                                     (bureau_merged['CREDIT_ACTIVE'] != 'Closed')].groupby('SK_ID_CURR').agg(aggregations_CREDIT_ACTIVE)
bureau_merged_aggregated_credit_rest.columns = ['_'.join(ele).upper() + 'CREDIT_ACTIVE_REST' for ele in bureau_merged_aggregated_credit_rest.columns]
#merging with other categories
bureau_merged_aggregated_credit = bureau_merged_aggregated_credit.merge(bureau_merged_aggregated_credit_rest, on = 'SK_ID_CURR', how = 'outer')
# -> đoạn này encoding các biến categorical ở bảng bureau -> thành tên bureau_merged < bảng này ở bài t đã merge bureau_balance nhưng có vẻ k ảnh hưởng>

#aggregating the bureau_merged over all the columns
bureau_merged_aggregated = bureau_merged.drop('SK_ID_BUREAU', axis = 1).groupby('SK_ID_CURR').agg('mean')
bureau_merged_aggregated.columns = [ele + '_MEAN_OVERALL' for ele in bureau_merged_aggregated.columns]
#merging it with aggregates over categories
bureau_merged_aggregated = bureau_merged_aggregated.merge(bureau_merged_aggregated_credit, on = 'SK_ID_CURR', how = 'outer')


