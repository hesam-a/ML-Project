import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.impute import SimpleImputer

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

# training set
dataset = pd.read_csv("SMRT_fixed.csv",low_memory=False)
dataset = miss_val_fixer(dataset)
data = dataset[:,2:]
norm_data = normalizer(data)

train_X = norm_data[:60000,:-1]
train_Y = norm_data[:60000,-1:]

# test set
test_X = norm_data[:10000,:-1]
test_Y = norm_data[:10000,-1:]

# NN model
model = Sequential()
# model.add(Dense(500, input_dim=1576, kernel_initializer='normal', activation='relu'))
# model.add(Dense(100, input_dim=1576, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(1 , kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X,train_Y,epochs=50, validation_data=(test_X,test_Y))

# model evaluatation
#results = nn_model.evaluate(test_X, test_Y)
#print("test loss, test acc:", round(results,2))
