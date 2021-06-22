import scipy.io
import scipy
import numpy as np
import numpy.matlib
import pickle
import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
reg = pickle.load(open('saved_model/3layers_MLP.sav', 'rb'))
scaler = pickle.load(open('saved_model/scaler_3layers.sav', 'rb'))


subject_name='INN-104-RWB'
x_test = scipy.io.loadmat(subject_name+ '/processed/ROI_DL.mat')
TestSig = x_test['Signal']
TestPredict=reg.predict(TestSig)
TestPredict = scaler.inverse_transform(TestPredict)
data = {}
data['DLprediction'] = TestPredict
directory=subject_name+'/processed/prediction/'
scipy.io.savemat(subject_name+ '/processed/prediction_DL.mat',data)
print('saved pred')