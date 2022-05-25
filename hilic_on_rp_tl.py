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

def standardize1(train):
    scaler = StandardScaler()
    std_train = scaler.fit_transform(train)
    return std_train

def standardize2(train,valid):
    scaler = StandardScaler()
    std_train = scaler.fit_transform(train)
    std_valid = scaler.transform(valid)
    return std_train,std_valid


nn_rp = tf.keras.models.Sequential([
    layers.Dense(1000, activation="relu", kernel_regularizer=l2(0.0001),input_shape = (77977, 1567)),#kernel_initializer="he_normal"
    layers.Dense(500,  activation="relu", kernel_regularizer=l2(0.0001)),#)input_shape = tal_X.shape),#kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(200,  activation="relu", kernel_regularizer=l2(0.0001)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(100,  activation="relu", kernel_regularizer=l2(0.0001)),#input_shape = metlin_X.shape[1:],kernel_regularizer=l2(0.0001)),#kernel_initializer="he_normal"
    layers.Dense(50 ,  activation="relu", kernel_regularizer=l2(0.0001)),
    layers.Dense(1)])

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
nn_rp.compile(loss="mean_squared_error", optimizer=opt)

# RP dataset
rp_dataset = pd.read_csv("Tal_olddd_dataset.csv",low_memory=False)
rp_dataset = rp_dataset[rp_dataset.rt > 300.0]
rp_dataset = miss_val_fixer(rp_dataset)

rp_data    = rp_dataset[:,3:]
rp_X       = standardize1(rp_data[:,:-1])
rp_Y       = rp_data[:,-1]

# HILIC dataset
hilic_dataset = pd.read_csv("Tal_smalll_dataset.csv",low_memory=False)
hilic_dataset = hilic_dataset.drop("NAME", axis=1)
hilic_dataset = miss_val_fixer(hilic_dataset)
hilic_data    = hilic_dataset[:,2:]

# training the entire RP data
#for i in range(16):
#    history_rp = nn_rp.fit(rp_X, rp_Y,epochs=20,batch_size=35,shuffle=True)
#    g = nn_rp.predict(rp_X)
#    h = np.reshape(rp_Y,(77977,1))
#    slope_tcn, intcept_tcn, rval_tcn, pval_tcn, stderr_tcn= stats.mstats.linregress(g,h)
#    r2_val_file = open("r2_rp_train_tl.txt",'a')
#    r2_val_file.write("{:2}{:15.6f} \n".format(i,rval_tcn))
#    nn_rp.save(f"nn_rp_{i}.h5")
#    r2_val_file.close()

split = 0.
for i in range(19):
    split += 0.05

    for j in range(20):

        train_set, test_set = train_test_split(hilic_data, test_size=round(split,2))#, random_state=42)    
        hilic_X, hilic_tstX = standardize2(train_set[:,:-1],test_set[:,:-1])
        # multiplied y (rt values) by 60 to convert them into second in order to be compatible with rp data
        hilic_Y             = train_set[:,-1]*60
        hilic_tstY          = test_set[:,-1]*60
        
        rp_model    = keras.models.load_model("nn_rp_15.h5")
        hilic_on_rp = keras.models.Sequential(rp_model.layers[:-1])
        hilic_on_rp.add(keras.layers.Dense(1, activation="relu"))#,input_shape = rp_X.shape[1:]))
#        hilic_input = keras.Input(shape=(hilic_tstX.shape[1:]))
        rp_clone    = keras.models.clone_model(rp_model)#,input_tensors = hilic_input)
        rp_clone.set_weights(rp_model.get_weights())
    
        for layer in hilic_on_rp.layers[:-1]:
            layer.trainable = False

        hilic_on_rp.compile(loss="mean_squared_error", optimizer="adam")
        history_1 = hilic_on_rp.fit(hilic_tstX, hilic_tstY, epochs=4, validation_data=(hilic_X, hilic_Y))

        for layer in hilic_on_rp.layers[:-1]:
            layer.trainable = True

        #optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-2
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        hilic_on_rp.compile(loss="mean_squared_error", optimizer=opt)
        history_2 = hilic_on_rp.fit(hilic_tstX, hilic_tstY, epochs=16, validation_data=(hilic_X, hilic_Y))
    
        nY = int(str(hilic_Y.shape).replace('(','').replace(')','').replace(',',''))
    
        a = hilic_on_rp.predict(hilic_X)
        b = np.reshape(hilic_Y,(nY,1))
        slope_tcn, intcept_tcn, rval_tcn, pval_tcn, stderr_tcn= stats.mstats.linregress(a,b)
        r2=f"hilic_on_rp_{round(split,2)}.txt"
        r2_val_file = open(r2,'a')
        r2_val_file.write("{:2}{:15.6f} \n".format(j,rval_tcn))
        r2_val_file.close()
    

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

