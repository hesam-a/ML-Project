import pandas as pd
import numpy as np
import os
np.set_printoptions(precision=10, suppress=True)
from scipy import stats

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import Sequential
#from keras.layers import Dense, Flatten, Convolution1D
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def miss_val_fixer(data):
    imp = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
    imp = imp.fit(data)
    fixed_data = imp.transform(data)
    return fixed_data

#def standardize(train,valid):
#    scaler = StandardScaler()
#    std_train = scaler.fit_transform(train)
#    std_valid = scaler.transform(valid)
#    return std_train,std_valid

def standardize(train):
    scaler = StandardScaler()
    std_train = scaler.fit_transform(train)
    return std_train


nn_small_tal = tf.keras.models.Sequential([
    layers.Dense(500, activation="relu", kernel_regularizer=l2(0.0001),input_shape =(533,1567)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(200, activation="relu", kernel_regularizer=l2(0.0001)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(100, activation="relu", kernel_regularizer=l2(0.0001)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(1)])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
nn_small_tal.compile(loss="mean_squared_error", optimizer=opt)


# RP dataset
dataset = pd.read_csv("Tal_olddd_dataset.csv",low_memory=False)
dataset = dataset[dataset.rt < 600.0]
dataset = miss_val_fixer(dataset)

data = dataset[:,3:]
tal_X = standardize(data[:,:-1])
tal_Y = data[:,-1]


# HILIC dataset
small_dataset = pd.read_csv("Tal_smalll_dataset.csv",low_memory=False)
small_dataset = small_dataset.drop("NAME", axis=1)
small_dataset = miss_val_fixer(small_dataset)
small_data = small_dataset[:,2:]

small_tal_X = standardize(small_data[:,:-1])
small_tal_Y = small_data[:,-1]


for i in range(1000):

    history_nn_small_tal = nn_small_tal.fit(small_tal_X, small_tal_Y,epochs=20,batch_size=35,shuffle=True)
    a = nn_small_tal.predict(tal_X)
    b = np.reshape(tal_Y,(8945,1))
    slope_tcn, intcept_tcn, rval_tcn, pval_tcn, stderr_tcn= stats.mstats.linregress(a,b)
    r2_val_file = open("r2_hilic_on_rp_10_vals.txt",'a')
    r2_val_file.write("{:2}{:13.6f} \n".format(i,rval_tcn))

r2_val_file.close()


#x_cn = range(0,30)
#y_cn = slope_cn * x_cn + intcept_cn
#labl_cn ='y = '+str(round(slope_cn))+' x + ('+str(round(intcept_cn,2))+')'
#plt.plot(x_cn,y_cn, color='red', label=labl_cn)
#plt.xlim([0,30])
#plt.ylim([0,30])
#plt.scatter(a,b,s=8,label='p-value= '+str(pval_cn)+", R2 = "+str(round(rval_cn**2,2))+'\n'+'Y mean: '+str(round(b.mean(),2))+'\npred mean: '+str(round(a.mean(),2)))
#plt.xlabel("Prediction")
#plt.ylabel("train Y")
#plt.legend()
#plt.show()
