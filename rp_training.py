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

def standardize(train,valid):
    scaler = StandardScaler()
    std_train = scaler.fit_transform(train)
    std_valid = scaler.transform(valid)
    return std_train,std_valid

#def standardize(train):
#    scaler = StandardScaler()
#    std_train = scaler.fit_transform(train)
#    return std_train


nn_tal = tf.keras.models.Sequential([
    layers.Dense(1000, activation="relu", kernel_regularizer=l2(0.0001),input_shape = (58482, 1567)),#kernel_initializer="he_normal"
    layers.Dense(500,  activation="relu", kernel_regularizer=l2(0.0001)),#)input_shape = tal_X.shape),#kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(200,  activation="relu", kernel_regularizer=l2(0.0001)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(100,  activation="relu", kernel_regularizer=l2(0.0001)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(50 ,  activation="relu", kernel_regularizer=l2(0.0001)),
    layers.Dense(1)])

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
nn_tal.compile(loss="mean_squared_error", optimizer=opt)


# Tal old dataset
dataset = pd.read_csv("Tal_olddd_dataset.csv",low_memory=False)
dataset = dataset[dataset.rt > 360.0]
dataset = miss_val_fixer(dataset)

data = dataset[:,3:]
#tal_X = standardize(data[:,:-1])
#tal_Y = data[:,-1]


for i in range(1000):

    train_set, test_set = train_test_split(data, test_size=0.25)#, random_state=42)    
    tal_X, tal_tst_X = standardize(train_set[:,:-1],test_set[:,:-1])
    tal_Y = train_set[:,-1]
    tal_tst_Y = test_set[:,-1]
    history_nn_tal = nn_tal.fit(tal_X, tal_Y, validation_data=(tal_tst_X,tal_tst_Y),epochs=20,batch_size=35,shuffle=True)
    g = nn_tal.predict(tal_tst_X)/60.
    h = np.reshape(tal_tst_Y,(19494,1))/60.
    slope_tcn, intcept_tcn, rval_tcn, pval_tcn, stderr_tcn= stats.mstats.linregress(g,h)
    r2_val_file = open("r2_vals.txt",'a')
    r2_val_file.write("{:2}{:15.6f} \n".format(i,rval_tcn))

r2_val_file.close()

#    e = nn_tal.predict(tal_X)/60.
#    f = np.reshape(tal_Y,(60026,1))/60.
#    slope_cn, intcept_cn, rval_cn, pval_cn, stderr_cn= stats.mstats.linregress(e,f)
#    p_val_file.write("{:2}{:15.6f}".format(i,pval_cn))

#x_cn = range(0,30)
#y_cn = slope_cn * x_cn + intcept_cn
#labl_cn ='y = '+str(round(slope_cn))+' x + ('+str(round(intcept_cn,2))+')'
#plt.plot(x_cn,y_cn, color='red', label=labl_cn)
#plt.xlim([0,30])
#plt.ylim([0,30])
#plt.scatter(e,f,s=8,label='p-value= '+str(pval_cn)+", R2 = "+str(round(rval_cn**2,2))+'\n'+'Y mean: '+str(round(f.mean(),2))+'\npred mean: '+str(round(e.mean(),2)))
#plt.xlabel("Prediction")
#plt.ylabel("train Y")
#plt.legend()
#plt.show()

#x_tcn = range(0,30)
#y_tcn = slope_tcn * x_tcn + intcept_tcn
#labl_tcn ='y = '+str(round(slope_tcn,2))+' x + '+str(round(intcept_tcn,2))
#plt.plot(x_tcn,y_tcn, color='red',label=labl_tcn)
#plt.xlim([0,30])
#plt.ylim([0,30])
#plt.scatter(g,h,s=8,label='p-value= '+str(pval_tcn)+", R2 = "+str(round(rval_tcn**2,2))+'\n'+'Y mean: '+str(round(h.mean(),2))+'\npred mean: '+str(round(g.mean(),2)))
#plt.xlabel("Prediction")
#plt.ylabel("Test Y")
#plt.legend()
#plt.show()
#
