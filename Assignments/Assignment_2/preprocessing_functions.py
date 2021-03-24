import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    df = pd.read_csv(df_path)
    return df



def divide_train_test(df, target):
    # Function divides data set in train and test
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(df.drop(target,axis=1),df[target],test_size=0.2, random_state=0)
    return Xtrain,Xtest,Ytrain,Ytest



def extract_cabin_letter(df, var):
    # captures the first letter
    return df[var].str[0] 



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    return np.where(df[var].isnull(), 1, 0)


    
def impute_na(df,val='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    for c in df.columns:
        if val == 'Missing':
            df[c].fillna(val,inplace=True)
        else:
            df[c].fillna(df[c].mode()[0],inplace=True)
    return df



def remove_rare_labels(df,var,rare_perc):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    for cat in var:
        tmp = df.groupby(cat)[cat].count() / len(df)
        labels =  tmp[tmp > rare_perc].index
        df[cat] = np.where(df[cat].isin(labels), df[cat], 'Rare')
    return df



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    
    df = df.copy()
    for cat in var:
        df = pd.concat([df,pd.get_dummies(df[cat], prefix=cat, drop_first=True)], axis=1)
    df.drop(var,axis=1,inplace=True)
    return df



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    missing_vars = [var for var in dummy_list if var not in df.columns]
    
    if len(missing_vars) == 0:
        print('All dummies were added')
    else:
        print(missing_vars)
        for var in missing_vars:
            df[var] = 0
    
    return df
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler,output_path)
    return scaler
  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    return scaler.transform(df)
    



def train_model(df, target, output_path):
    # train and save model
    clf = LogisticRegression(C=0.0005, random_state=0)
    clf.fit(df,target)
    joblib.dump(clf,output_path)



def predict(df, model):
    # load model and get predictions
    clf = joblib.load(model)
    return clf.predict(df)

