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


# # impute categorical variables
xtrain[config.CATEGORICAL_VARS]=pf.impute_na(xtrain[config.CATEGORICAL_VARS],'Missing')

# # impute numerical variable
xtrain[config.NUMERICAL_TO_IMPUTE]=pf.impute_na(xtrain[config.NUMERICAL_TO_IMPUTE],'Numerical')


# # Group rare labels
for var in config.CATEGORICAL_VARS:
    xtrain[var] = pf.remove_rare_labels(xtrain, var,config.FREQUENT_LABELS[var])


# # encode categorical variables
xtrain = pf.encode_categorical(xtrain,config.CATEGORICAL_VARS)


# # check all dummies were added
xtrain = pf.check_dummy_variables(xtrain,config.DUMMY_VARIABLES)


# # train scaler and save
scaler = pf.train_scaler(xtrain,config.OUTPUT_SCALER_PATH)


# # scale train set
xtrain = pf.scale_features(xtrain,config.OUTPUT_SCALER_PATH)

# train model and save
pf.train_model(xtrain,ytrain,config.OUTPUT_MODEL_PATH)


print('Finished training')