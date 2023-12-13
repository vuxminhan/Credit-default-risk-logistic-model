# Now you can import utils
from src.helpers.utils import *
from src.features.01_application_train import 

def pre_processing_and_combine():

    
    with timer("Process application train"):
        df = application_train()
        print("application train & test shape:", df.shape)
        
    
    with timer("Bureau and Bureau Balance"):
        bureau_and_bb_agg = bureau_bb()
        print("Bureau and Bureau Balance:", bureau_and_bb_agg.shape)
        
    with timer("Installment Payments"):
        agg_list_previous_application, ins_agg = installments_payments()
        print("Installment Payments:", ins_agg.shape)    
    
    with timer("Pos Cash Balance"):
        agg_list_previous_application, pos_agg = pos_cash_balance(agg_list_previous_application)
        print("Pos Cash Balance:", pos_agg.shape)  
        
    
    with timer("Credit Card Balance"):
        CCB_agg = credit_card_balance()
        print("Credit Card Balance:", CCB_agg.shape) 
    
    with timer("previous_application"):
        prev_agg_list, df_prev = previous_application(agg_list_previous_application)
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
    
    