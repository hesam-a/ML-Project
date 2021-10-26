import pandas as pd
import numpy as np
np.set_printoptions(precision=10, suppress=True)

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

def standardize(train,valid):
    scaler = StandardScaler()
    std_train = scaler.fit_transform(train)
    std_valid = scaler.transform(valid)
    return std_train,std_valid

def normalizer(data):
    layer = preprocessing.Normalization(axis=None)
    layer.adapt(data)
    normalized_data = layer(data)
    array_data = np.array(normalized_data) 
    return array_data

def nn_model():
    model = Sequential()
    model.add(Dense(500, activation='relu', kernel_regularizer=l2(0.0001),input_shape=train_X.shape[1:])
    model.add(Dense(200, kernel_regularizer=l2(0.0001), activation='relu')) # kernel_initializer='he_normal',
    model.add(Dense(100, kernel_regularizer=l2(0.0001), activation='relu')) # kernel_initializer='he_normal',
    model.add(Dense(50 , kernel_regularizer=l2(0.0001), activation='relu')) # kernel_initializer='he_normal'
    model.add(Dense(20, kernel_regularizer=l2(0.0001), activation='relu')) # kernel_initializer='he_normal',
    model.add(Dense(1 )# kernel_initializer='he_normal'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer='opt',metrics=['mae'])
    return model

# dataset
dataset = pd.read_csv("SMRT_fixed.csv",low_memory=False)
dataset = miss_val_fixer(dataset)
data = dataset[:,2:]
train_set, test_set = train_test_split(data, test_size=0.25, random_state=42)

train_X, test_X = standardize(train_set[:,:-1],test_set[:,:-1])
train_Y = train_set[:,-1]/10
test_Y = test_set[:,-1]/10

# NN model
estimator = KerasRegressor(build_fn=nn_model, epochs=20, verbose=0)
kfold = KFold(n_splits=20)
results = cross_val_score(estimator,train_X, train_Y, cv=kfold)
print("mean: %.2f (%.2f) std dev" % (results.mean(), results.std()))
