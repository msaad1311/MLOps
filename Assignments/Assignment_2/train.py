import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
xtrain,xtest,ytrain,ytest = pf.divide_train_test(df,config.TARGET)


# # get first letter from cabin variable
xtrain['cabin'] = pf.extract_cabin_letter(xtrain,'cabin')
xtest['cabin'] = pf.extract_cabin_letter(xtest,'cabin')


# # impute categorical variables
xtrain[config.CATEGORICAL_VARS]=pf.impute_na(xtrain[config.CATEGORICAL_VARS],'Missing')
xtest[config.CATEGORICAL_VARS]=pf.impute_na(xtest[config.CATEGORICAL_VARS],'Missing')

# # impute numerical variable
xtrain[config.NUMERICAL_TO_IMPUTE]=pf.impute_na(xtrain[config.NUMERICAL_TO_IMPUTE],'Numerical')
xtest[config.NUMERICAL_TO_IMPUTE]=pf.impute_na(xtest[config.NUMERICAL_TO_IMPUTE],'Numerical')


# # Group rare labels
xtrain=pf.remove_rare_labels(xtrain,config.CATEGORICAL_VARS,0.05)
xtest=pf.remove_rare_labels(xtest,config.CATEGORICAL_VARS,0.05)


# # encode categorical variables
xtrain = pf.encode_categorical(xtrain,config.CATEGORICAL_VARS)
xtest = pf.encode_categorical(xtest,config.CATEGORICAL_VARS)

# # check all dummies were added
xtrain = pf.check_dummy_variables(xtrain,config.DUMMY_VARIABLES)


# # train scaler and save
scaler = pf.train_scaler(xtrain,config.OUTPUT_SCALER_PATH)


# # scale train set
xtrain = pf.scale_features(xtrain,config.OUTPUT_SCALER_PATH)
xtest = pf.scale_features(xtest,config.OUTPUT_SCALER_PATH)

# train model and save
pf.train_model(xtrain,ytrain,config.OUTPUT_MODEL_PATH)


print('Finished training')