import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import svm

def miss_val_fixer(data):
    imp = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
    imp = imp.fit(data)
    fixed_data = imp.transform(data)
    return fixed_data

def normalizer(data):
    layer = preprocessing.Normalization(axis=None)
    layer.adapt(data)
    normalized_data = layer(data)
    array_data = np.array(normalized_data) 
    return array_data

def baseline_model():
    model = Sequential()
#    model.add(Dense(500, input_dim=1576, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(100, input_dim=1576, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1 , kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

train_set = pd.read_csv("SMRT_fixed.csv", nrows=60000,low_memory=False)
train_set = miss_val_fixer(train_set)
train_set = train_set[:,2:]
norm_set = normalizer(train_set) 

train_X = norm_set[:,:-1]
train_Y = norm_set[:,-1:]

# normalizing training X and Y
#norm_X = normalizer(train_X)
#norm_Y = normalizer(train_Y)

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, verbose=0)
kfold = KFold(n_splits=40)
results = cross_val_score(estimator,train_X, train_Y, cv=kfold)
print("mean: %.2f (%.2f) std dev" % (results.mean(), results.std()))

#prediction = estimator.predict(X_test)


#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10)
#results = cross_val_score(pipeline, train_X, train_Y, cv=kfold)
#print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
