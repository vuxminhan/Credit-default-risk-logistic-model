import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Now you can import utils
from src.helpers.utils import *

def application_train(df,test_df):
    
    df = pd.concat([df,test_df]).reset_index()
    print(df['TARGET'].isnull().sum())
#     df = df[df['CODE_GENDER'] != 'XNA']
    df.loc[df['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = 'F'
    print(df['TARGET'].isnull().sum())
    lbe = LabelEncoder()

    for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[col] = lbe.fit_transform(df[col])
    
    df['TOTALAREA_MODE'] = df['TOTALAREA_MODE'] ** (1/3)
    
    education_type_mapping = {
        'Lower secondary': 1,
        'Secondary / secondary special': 2,
        'Incomplete higher': 3,
        'Higher education': 4,
        'Academic degree': 5
    }
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(education_type_mapping)
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype('int')

    
    def group_organizations(org_type):
        if 'Trade' in org_type:
            return 'Trade'
        elif 'Industry' in org_type:
            return 'Industry'
        elif 'Business' in org_type:
            return 'Business Entity'
        elif 'Transport' in org_type:
            return 'Transport'
        else:
            return org_type
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].apply(group_organizations)
    
    med_income = df.groupby(['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE'])['AMT_INCOME_TOTAL'].transform('median')
    med_income2 = df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
    df['income_ratio'] = df['AMT_INCOME_TOTAL'] / med_income
    df['income_ratio2'] = df['AMT_INCOME_TOTAL'] / med_income2
    df['true_annuity_div_income'] = df['AMT_ANNUITY'] / med_income
    df['true_annuity_div_income2'] = df['AMT_ANNUITY'] / med_income2
    df['true_income_div_totalarea'] = med_income / df['TOTALAREA_MODE']
    df['true_income_div_totalarea2'] = med_income2 / df['TOTALAREA_MODE']

    df['TOTAL_ENQUIRIES_CREDIT_BUREAU'] = df[['AMT_REQ_CREDIT_BUREAU_DAY',
                                          'AMT_REQ_CREDIT_BUREAU_HOUR',
                                          'AMT_REQ_CREDIT_BUREAU_WEEK',
                                          'AMT_REQ_CREDIT_BUREAU_MON',
                                          'AMT_REQ_CREDIT_BUREAU_QRT',
                                          'AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)
    
    df['PCTG_ENQUIRIES_HOUR'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] / df['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    df['PCTG_ENQUIRIES_DAY'] = df['AMT_REQ_CREDIT_BUREAU_DAY'] / df['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    df['PCTG_ENQUIRIES_WEEK'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] / df['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    df['PCTG_ENQUIRIES_MON'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / df['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    df['PCTG_ENQUIRIES_QRT'] = df['AMT_REQ_CREDIT_BUREAU_QRT'] / df['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    df['PCTG_ENQUIRIES_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'] / df['TOTAL_ENQUIRIES_CREDIT_BUREAU']
        
    columns_to_remove = ['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_WEEK']
    df.drop(columns=columns_to_remove, inplace=True)
    print(df['TARGET'].isnull().sum())
    df = pd.get_dummies(df, dummy_na = True)
    print(df['TARGET'].isnull().sum())
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['NEW_PAYMENT_RATE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['DEBT_BURDEN_PER_WORKING_DAY'] = df['NEW_PAYMENT_RATE'] / df['DAYS_EMPLOYED']
    df['DEBT_BURDEN_PER_LIFE_DAY'] = df['NEW_PAYMENT_RATE'] / df['DAYS_BIRTH']
    # represent the ratio of the difference between the loan amount and the value of goods compared to the value of the goods
    df['CREDIT_GOODS_PRICE_RATIO1'] = (df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']) /  df['AMT_GOODS_PRICE']
    # represent the ratio of the difference between the loan amount and the value of goods compared to the loan amount
    df['CREDIT_GOODS_PRICE_RATIO2'] = (df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']) /  df['AMT_CREDIT']
    df['GOODS_PRICE_TO_ANNUITY_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_GOODS_PRICE']
    
    df['DIFF_OBS_30_CNT_SOCIAL_CIRCLE_OBS_60_CNT_SOCIAL_CIRCLE'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] - df['OBS_60_CNT_SOCIAL_CIRCLE']
    
    df['DIFF_DEF_30_CNT_SOCIAL_CIRCLE_DEF_60_CNT_SOCIAL_CIRCLE'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] - df['DEF_60_CNT_SOCIAL_CIRCLE']
    
    
    missing_columns = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                   'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
                   'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
                   'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']
    df['MISSING_GRADINGS'] = df[missing_columns].isna().sum(axis=1)
    df['RATIO_AMT_GOODS_PRICE_TO_LIVINGAREA_AVG'] = df['AMT_GOODS_PRICE'] / df['LIVINGAREA_AVG']
    df['RATIO_AMT_GOODS_PRICE_TO_LANDAREA_AVG'] = df['AMT_GOODS_PRICE'] / df['LANDAREA_AVG']
    df['RATIO_AMT_GOODS_PRICE_TO_FLOORSMAX_AVG_AVG'] = df['AMT_GOODS_PRICE'] / df['FLOORSMAX_AVG']
    df['RATIO_AMT_GOODS_PRICE_TO_LIVINGAPARTMENTS_AVG'] = df['AMT_GOODS_PRICE'] / df['LIVINGAPARTMENTS_AVG']
    df['RATIO_AMT_GOODS_PRICE_TO_YEARS_BUILD_AVG'] = df['AMT_GOODS_PRICE'] / df['YEARS_BUILD_AVG'] 
    # Create new features from EXT source columns
#     df['EXT_SOURCES_WEIGHTED'] = df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 4
#     for function_name in ['min', 'max', 'sum', 'mean']:
#         df['EXT_SORCES_{}'.format(function_name)] = getattr(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], function_name)(axis=1)
#     df['EXT_sOURCE_NANMEDIAN'] = np.nanmedian(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    
    df['EXT_SOURCE_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1, skipna=True)
    df['EXT_SOURCE_PROD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].prod(axis=1, skipna=True)
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1, skipna=True)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1, skipna=True)
    df['EXT_SOURCE_MISSING_VALUES'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].isna().sum(axis=1)

    
    # Converted so 0 is close to 23
    df['sin_HOUR_APPR_PROCESS_START'] = np.sin(2 * np.pi * df['HOUR_APPR_PROCESS_START'] / 24)
    df['cos_HOUR_APPR_PROCESS_START'] = np.cos(2 * np.pi * df['HOUR_APPR_PROCESS_START'] / 24)
    df.drop(columns=['HOUR_APPR_PROCESS_START'], inplace=True)
    
    # numerical transformation
    df['REGION_POPULATION_RELATIVE'] = np.sqrt(df['REGION_POPULATION_RELATIVE'])
    df['APARTMENTS_AVG'] = np.log1p(50 * df['APARTMENTS_AVG'])
    df['YEARS_BEGINEXPLUATATION_AVG'] = df['YEARS_BEGINEXPLUATATION_AVG'] ** 30
    df['YEARS_BUILD_AVG'] = df['YEARS_BUILD_AVG'] ** 3
    df['COMMONAREA_AVG'] = df['COMMONAREA_AVG'] ** (-1/200)
    df['ELEVATORS_AVG'] = df['ELEVATORS_AVG'] ** (1/40)
    df['ENTRANCES_AVG'] = df['ENTRANCES_AVG'] ** (1/3)
    df['FLOORSMAX_AVG'] = df['FLOORSMAX_AVG'] ** (1/2.5)
    df['FLOORSMIN_AVG'] = df['FLOORSMIN_AVG'] ** (1/2.2)
    df['LANDAREA_AVG'] = df['LANDAREA_AVG'] ** (1/5)
    df['LIVINGAPARTMENTS_AVG'] = df['LIVINGAPARTMENTS_AVG'] ** (1/3)
    df['LIVINGAREA_AVG'] = df['LIVINGAREA_AVG'] ** (1/3.5)
    df['NONLIVINGAPARTMENTS_AVG'] = df['NONLIVINGAPARTMENTS_AVG'] ** (1/7)
    df['NONLIVINGAREA_AVG'] = df['NONLIVINGAREA_AVG'] ** (1/5)
    df['OBS_30_CNT_SOCIAL_CIRCLE'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] ** (1/7)
    df['DEF_30_CNT_SOCIAL_CIRCLE'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] ** (1/7)
    df['OBS_60_CNT_SOCIAL_CIRCLE'] = df['OBS_60_CNT_SOCIAL_CIRCLE'] ** (1/7)
    df['DEF_60_CNT_SOCIAL_CIRCLE'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] ** (1/7)
    
    missing_columns = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                   'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
                   'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
                   'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']

    df['MISSING_GRADINGS'] = df[missing_columns].isna().sum(axis=1)
    # Reliability in customer city or region of residence
    df['RELIABILITY_IN_CUSTOMER_CITY'] = df['REG_CITY_NOT_LIVE_CITY'] + df['REG_CITY_NOT_WORK_CITY'] + df['REG_REGION_NOT_LIVE_REGION'] + df['REG_REGION_NOT_WORK_REGION'] + df['LIVE_CITY_NOT_WORK_CITY'] + df['LIVE_REGION_NOT_WORK_REGION']
    df['SUM_CONTACTS'] = df['FLAG_MOBIL'] + df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] + df['FLAG_CONT_MOBILE'] + df['FLAG_PHONE'] + df['FLAG_EMAIL']
    
    
    #     Create new features from document columns
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['HAS_DOCUMENT'] = df[docs].max(axis=1)
    # Drop most flag document columns
    drop_list = []
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    
    df.drop("index", axis = 1, inplace =  True)

    df.columns = pd.Index(["APP_" + col for col in df.columns.tolist()])

    df.rename(columns={"APP_SK_ID_CURR":"SK_ID_CURR"}, inplace = True)

    df.rename(columns={"APP_TARGET":"TARGET"}, inplace = True)
    print(df['TARGET'].isnull().sum())
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/dseb63_application_train.csv')
    test_df = pd.read_csv('data/raw/dseb63_application_test.csv')
    
    processed_df = application_train(df,test_df)
    processed_df.to_csv('data/interim/df.csv', index=False)