import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import titanic_pipe
import config



def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv('titanic.csv')
    # divide train and test
    xtrain,xtest,ytrain,ytest= train_test_split(data.drop(config.TARGET,axis=1),data[config.TARGET],test_size=0.2,random_state=0)
    # fit pipeline
    titanic_pipe.fit(xtrain,ytrain)
    # save pipeline
    joblib.dump(titanic_pipe,config.PIPELINE_NAME)

if __name__ == '__main__':
    run_training()
