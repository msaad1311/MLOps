from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    ('categorical_imputer',pp.CategoricalImputer(config.CATEGORICAL_VARS)),
    ('missing_indicator',pp.MissingIndicator(config.NUMERICAL_VARS))
    ('numerical_imputer',pp.NumericalImputer(config.NUMERICAL_VARS)),
    ('cabin_extractor',pp.ExtractFirstLetter(config.CABIN)),
    
    )