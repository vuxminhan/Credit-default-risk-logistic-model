# Now you can import utils
import os
import sys
import pickle
sys.path.append(os.getcwd())
from src.helpers.utils import *
from src.features.simple_aggregates.application_train import application_train
from src.features.simple_aggregates.bureau_bureau_balance import bureau_bb
from src.features.simple_aggregates.instalment_payment import installments_payments
from src.features.simple_aggregates.pos_cash_balance import pos_cash_balance
from src.features.simple_aggregates.credit_card_balance import credit_card_balance
from src.features.simple_aggregates.previous_application import previous_application

def pre_processing_and_combine():

    
    with timer("Process application train"):
        train_df = pd.read_csv('data/raw/dseb63_application_train.csv')
        test_df = pd.read_csv('data/raw/dseb63_application_test.csv')
    
        df = application_train(train_df, test_df)
        print("application train & test shape:", df.shape)
        
    
    with timer("Bureau and Bureau Balance"):
        bureau = pd.read_csv('data/raw/dseb63_bureau.csv')
        bereau_balance = pd.read_csv('data/raw/dseb63_bureau_balance.csv')
        bureau_and_bb_agg = bureau_bb(bereau_balance, bureau)
        print("Bureau and Bureau Balance:", bureau_and_bb_agg.shape)
        
    with timer("Installment Payments"):
        ins = pd.read_csv('data/raw/dseb63_installments_payments.csv')
        agg_list_previous_application, ins_agg = installments_payments(ins)
        print("Installment Payments:", ins_agg.shape)    
    
    with timer("Pos Cash Balance"):
        pos = pd.read_csv('data/raw/dseb63_POS_CASH_balance.csv')
        agg_list_previous_application, pos_agg = pos_cash_balance(agg_list_previous_application, pos)
        print("Pos Cash Balance:", pos_agg.shape)  
        
    
    with timer("Credit Card Balance"):
        CCB = pd.read_csv('data/raw/dseb63_credit_card_balance.csv')
        CCB_agg = credit_card_balance(CCB)
        print("Credit Card Balance:", CCB_agg.shape) 
    
    with timer("previous_application"):
        prev_app = pd.read_csv('data/raw/dseb63_previous_application.csv')
        prev_agg_list, df_prev = previous_application(prev_app, agg_list_previous_application)
        print("previous_application:", df_prev.shape) 
        
        
    with timer("All tables are combining"):
        df_prev_ins = df_prev.merge(ins_agg, how = 'left', on = 'SK_ID_PREV')
        df_prev_ins_pos = df_prev_ins.merge(pos_agg, how = 'left', on = 'SK_ID_PREV')
        df_prev_ins_pos_agg = df_prev_ins_pos.groupby("SK_ID_CURR").agg(prev_agg_list).reset_index()
        df_prev_ins_pos_agg.columns = pd.Index(["PREV_" + col[0] + "_" + col[1].upper() for col in df_prev_ins_pos_agg.columns.tolist()])
        df_prev_ins_pos_agg.rename(columns={"PREV_SK_ID_CURR_":"SK_ID_CURR"}, inplace = True)
        #main table with #prev_son
        df_prev_others = df.merge(df_prev_ins_pos_agg, how = 'left',on = 'SK_ID_CURR')
    
        #credit_card_balance
        df_prev_ins_pos_ccb = df_prev_others.merge(CCB_agg, how = 'left',on = 'SK_ID_CURR')
    
        #bureau_balance
        all_data = df_prev_ins_pos_ccb.merge(bureau_and_bb_agg, how = 'left',on = 'SK_ID_CURR')
        
        print("all_data process:", all_data.shape) 
    
    return all_data
    
with timer("Preprocessing Time"):
    all_data = pre_processing_and_combine()
    with open('data/processed/all_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)
