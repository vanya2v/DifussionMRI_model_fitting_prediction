import scipy.io
import scipy
import numpy as np
import numpy.matlib
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
print('Loading training set...')
x_train = scipy.io.loadmat('database_train_DL_fixdees_rician_SNR35.mat')
TrainSig = x_train['database_train_noisy']
TrainParam = x_train['params_train_noisy']
print('Setting up the DL model...')
reg = MLPRegressor(hidden_layer_sizes=(150, 150, 150),  activation='relu', solver='adam', alpha=0.001, batch_size=100, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.20, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(TrainParam)
TrainParam=scaler.transform(TrainParam)
print('Training the DL model...')
reg.fit(TrainSig,TrainParam)
print('Saving the trained DL model...')
filename = 'saved_model/3layers_MLP.sav'
pickle.dump(reg, open(filename, 'wb'))
filename = 'saved_model/scaler_3layers.sav'
pickle.dump(scaler, open(filename, 'wb'))


print('DONE')
