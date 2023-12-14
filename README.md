# Data-Preparation-Group-5

### This is the notebook for final EDA project by Group 5, DSEB K63, NEU
  
  * Vu Minh An ( Group leader )
  * Nhan Yen Trang
  * Le Ngoc Anh
  
  [Link to Kaggle notebook](https://www.kaggle.com/competitions/home-credit-default-risk)

  [Link to Data File](https://drive.google.com/drive/folders/1CPOJGypsMPJE9vf6qe8UIDEi4JaIm__m?usp=sharing)
  
  [Link to Slide](https://www.canva.com/design/DAF22dYdw2w/nPVxsAnB8yGHXeCyvZDYNQ/edit)
  
Project Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    ├── EDA                <- Jupyter notebooks for Exploratory Data Analysis.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed using different methods
    │   │   ├── binned_numerical_features
    │   │   ├── lead_lag
    │   │   ├── nested_model
    │   │   ├── simple_aggregates
    │   │   └── slicing
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    └── src
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   ├── all_data.py <- Merge data for all methods
        │   ├── binned_numerical_features
        │   ├── lead_lag
        │   ├── nested_model
        │   ├── simple_aggregates
        │   └── slicing
        ├── helpers        <- Helper scripts and utilities.
        └── pipelines      <- Scripts for data processing and model training pipelines.
            ├── predict_pipeline.py
            └── train_pipeline.py

  ### Assigned work:
  * Nguyen Minh Duc: 
    - EDA:<br /> 
      * 02_Previous_application
      * 05_POS_Cash_balance 
    - Feature Engineer: <br />
      * 01_application_train
      * 02_Previous_application 
      * 05_POS_Cash_balance 
      * 08_Feature Selection Final (Merge data and Feature ) 
  * Nguyen Ngoc Dat:
    - EDA:<br />
      * 03_Bureau
      * 04_Bureau_balance
    - Feature Engineer: <br />
      * 03_Bureau
      * 04_Bureau_balance
      * 08_Feature Selection Final (handle outliners)
  * Le Quoc Minh:
    - EDA:<br />
      * 01_application_train 
    - Feature Engineer: <br />
      * 01_application_train 
      * 08_Feature Selection Final (handle outliners)
  * Le Huy Phong:
    - EDA: <br />
      * 06_instalment_payment
      * 07_credit_card_balance 
    - Feature Engineer: <br />
      * 06_instalment_payment
      * 07_credit_card_balance 
    - Slide
