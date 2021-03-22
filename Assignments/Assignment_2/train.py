import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
xtrain,xtest,ytrain,ytest = pf.divide_train_test(df,config.TARGET)


# get first letter from cabin variable
xtrain['cabin'] = pf.extract_cabin_letter(xtrain,'cabin')
xtest['cabin'] = pf.extract_cabin_letter(xtest,'cabin')


# impute categorical variables
xtrain[config.CATEGORICAL_VARS]=pf.impute_na(xtrain[config.CATEGORICAL_VARS],'Missing')
xtest[config.CATEGORICAL_VARS]=pf.impute_na(xtest[config.CATEGORICAL_VARS],'Missing')


# impute numerical variable
xtrain[config.NUMERICAL_TO_IMPUTE]=pf.impute_na(xtrain[config.NUMERICAL_TO_IMPUTE])
xtest[config.NUMERICAL_TO_IMPUTE]=pf.impute_na(xtest[config.NUMERICAL_TO_IMPUTE])


# Group rare labels
xtrain[config.CATEGORICAL_VARS]=pf.remove_rare_labels(xtrain,config.CATEGORICAL_VARS,0.05)
xtest[config.CATEGORICAL_VARS]=pf.remove_rare_labels(xtest,config.CATEGORICAL_VARS,0.05)


# encode categorical variables
xtrain[config.CATEGORICAL_VARS] = pf.encode_categorical(xtrain,config.CATEGORICAL_VARS)
xtest[config.CATEGORICAL_VARS] = pf.encode_categorical(xtest,config.CATEGORICAL_VARS)

# check all dummies were added



# train scaler and save



# scale train set



# train model and save



print('Finished training')