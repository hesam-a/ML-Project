import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

def standardize(train,valid):
    scaler = StandardScaler()
    std_train = scaler.fit_transform(train)
    std_valid = scaler.transform(valid)
    return std_train,std_valid

# dataset
dataset = pd.read_csv("SMRT_fixed.csv",low_memory=False)
dataset = miss_val_fixer(dataset)
data = dataset[:,2:]
train_set, test_set = train_test_split(data, test_size=0.25, random_state=42)


train_X, test_X = standardize(train_set[:,:-1],test_set[:,:-1])
train_Y = train_set[:,-1]/10
test_Y = test_set[:,-1]/10


model = tf.keras.models.Sequential([
#    layers.Dense(1000, activation="relu", kernel_initializer="he_normal",kernel_regularizer=l2(0.0001),input_shape=train_X.shape[1:], bias_regularizer=l2(0.0001)),
    layers.Dense(500 , activation="relu", kernel_regularizer=l2(0.0001)),# input_shape=train_X.shape[1:], kernel_initializer="he_normal"
    layers.Dense(200 , activation="relu", kernel_regularizer=l2(0.0001)),#input_shape=train_X.shape[1:]), kernel_initializer="he_normal"
    layers.Dense(100 , activation="relu", kernel_regularizer=l2(0.0001)),#input_shape=train_X.shape[1:]), kernel_initializer="he_normal"
    layers.Dense(50  , activation="relu", kernel_regularizer=l2(0.0001)),#input_shape=train_X.shape[1:]), kernel_initializer="he_normal"
    layers.Dense(20  , activation="relu", kernel_regularizer=l2(0.0001)),#input_shape=train_X.shape[1:]), kernel_initializer="he_normal"
    layers.Dense(1)]) #,kernel_initializer="he_normal")])

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="mean_squared_error", optimizer=opt,metrics=['mae'])
history = model.fit(train_X,train_Y, validation_data=(test_X,test_Y),epochs=300,batch_size = 35, shuffle=True)

plt.scatter(model.predict(train_X)/60,train_Y/60)
plt.ylabel("Prediction")
plt.xlabel("train Y")
plt.show()

plt.scatter(model.predict(test_X)/60,test_Y/60)
plt.ylabel("Prediction")
plt.xlabel("test Y")
plt.show()

# model evaluatation
#results = nn_model.evaluate(test_X, test_Y)
#print("test loss, test acc:", round(results,2))
