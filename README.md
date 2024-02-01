## Home Credit makes use of a variety of alternative data to predict their clients' repayment abilities
### This repo present a classical and explainable approach using Logistic Regression model combined with data engineering to predict client default 

  
  [Link to Kaggle notebook](https://www.kaggle.com/code/vuxminhan/home-credit-risk-project-final)

  [Link to Data File](https://drive.google.com/drive/folders/1CPOJGypsMPJE9vf6qe8UIDEi4JaIm__m?usp=sharing)

  [Link to Report](https://docs.google.com/document/d/1krnUQGKT-X_8ghDj6qU7QLddtmRhCt2c/edit?usp=sharing&ouid=104574113477201047691&rtpof=true&sd=true)
  
  [Link to Slide](https://drive.google.com/file/d/19yLAoLRlGoISugTgeAcIOiit7dN5-09S/view?usp=sharing)
  
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
