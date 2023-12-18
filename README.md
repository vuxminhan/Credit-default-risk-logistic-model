# Data-Preparation-Group-5

### This is the notebook for final EDA project by Group 5, DSEB K63, NEU
  
  * Vu Minh An ( Group leader )
  * Nhan Yen Trang
  * Le Ngoc Anh
  
  [Link to Kaggle notebook](https://www.kaggle.com/code/vuxminhan/home-credit-risk-project-final)

  [Link to Data File](https://drive.google.com/drive/folders/1CPOJGypsMPJE9vf6qe8UIDEi4JaIm__m?usp=sharing)

  [Link to Report](https://docs.google.com/document/d/18R21mAvv06bSUkkNcD8dnkoOMuYpm_5JIm3BxF-tPKc/edit?usp=sharing)
  
  [Link to Slide](https://www.canva.com/design/DAF3Ajr887k/vdd1MevYZenh68JN2iCk4g/edit?utm_content=DAF3Ajr887k&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
  
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
  * Vu Minh An:
    - Main idea
    - Modelling:<br />
      * Logistic Regression
    - Feature Engineer: <br />
      * Main tables: Application_train, Application_test
      * Feature Selection (Merge data and Feature/Feature importance using Lightgbm)
      * Weight of evidence/nested model/lead lag/simple aggregation
    - Report
  * Nhan Yen Trang:
    - EDA:<br />
      * Main tables: Application_train, Application_test
      * Bureau
      * Bureau_balance
    - Feature Engineer: <br />
       * Application_train|test
       * Bureau
       * Bureau_balance
       * credit_card_balance
    - Slides
  * Le Ngoc Anh:
    - EDA:<br />
      * previous_application
      * instalment_payment
      * POS_cash
      * credit_card_balance
    - Feature Engineer: <br />
      * previous_application
      * instalment_payment
      * POS_cash
