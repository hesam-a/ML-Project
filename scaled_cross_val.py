import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers.experimental import preprocessing

def miss_val_fixer(data):
    imp = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
    imp = imp.fit(data)
    fixed_data = imp.transform(data)
    return fixed_data

# sci-kit's normalizer
def norm(data):
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

def baseline_model():
    model = Sequential()
#    model.add(Dense(500)) #, input_dim=1578, use_bias=True, bias_initializer='zeros', activation='relu'))
#    model.add(Dense(100)) #,  activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1 , kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# load dataset
dataset = pd.read_csv("SMRT_fixed.csv",low_memory=False)
dataset = miss_val_fixer(dataset)
data = dataset[:,2:]
norm_data = norm(data)

train_X = norm_data[:60000,:-1]
train_Y = norm_data[:60000,-1:]

estimator = KerasRegressor(build_fn=baseline_model, epochs=50, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, train_X, train_Y, cv=kfold)
print("mean: %.2f (%.2f) std dev" % (results.mean(), results.std()))


#def scaler(data):
#    data_mean = np.mean(data)
#    X1 = data - data_mean
#    X2 = np.square(X1)
#    X2 = np.average(X2)
#    variance = np.sqrt(X2)
#    scaled_data = X1/variance
#    return scaled_data

## scaling train X and Y
#train_X_scaled = scaler(train_X)
#train_Y_scaled = scaler(train_Y)

#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10)
#results = cross_val_score(pipeline, train_X, train_Y, cv=kfold)
#print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

