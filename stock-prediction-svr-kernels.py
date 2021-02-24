TEST = 0.20 #0.2
WINDOW = 25 #25
C_VAL = 1e2 ##1e2
Y_VAL = 1e1 #1e1
FILE = 'aapl-1y.csv' #'aapl-1y.csv' 

import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

total_dates_train = []
total_dates_test = []
total_closes_train = []
total_closes_test = []

total_trained_closes_lin = []
total_trained_closes_pol = []
total_trained_closes_rbf = []

total_tested_closes_lin = []
total_tested_closes_pol = []
total_tested_closes_rbf = []

df = pd.read_csv(FILE)
mmscaler = MinMaxScaler()
stdscaler = StandardScaler()
pca = PCA(n_components = 1)

dates = df['Index'].tolist()
dates = np.reshape(dates, (len(dates), 1))
dates = mmscaler.fit_transform(dates)

closes = df['Adj Close'].tolist()
closes = np.reshape(closes, (len(closes), 1))
closes = mmscaler.fit_transform(closes)
closes = closes.ravel()

SIZE = len(dates)

for i in range(int((1-TEST)*WINDOW), SIZE, int(WINDOW)):

	dates_train, dates_test = dates[i-int((1-TEST)*WINDOW):i], dates[i:i+int(TEST*WINDOW)]
	closes_train, closes_test = closes[i-int((1-TEST)*WINDOW):i], closes[i:i+int(TEST*WINDOW)]	

	svr_lin = SVR(kernel = 'linear', C = C_VAL)
	svr_pol = SVR(kernel = 'poly', C = C_VAL, degree = 2, gamma = 'auto')
	svr_rbf = SVR(kernel = 'rbf', C = C_VAL, gamma = Y_VAL)
	
	svr_lin.fit(dates_train, closes_train)
	svr_pol.fit(dates_train, closes_train)
	svr_rbf.fit(dates_train, closes_train)

	trained_closes_lin = svr_lin.predict(dates_train)
	trained_closes_pol = svr_pol.predict(dates_train)
	trained_closes_rbf = svr_rbf.predict(dates_train)

	tested_closes_lin = svr_lin.predict(dates_test)
	tested_closes_pol = svr_pol.predict(dates_test)
	tested_closes_rbf = svr_rbf.predict(dates_test)

	total_dates_train.extend(dates_train)
	total_dates_test.extend(dates_test)
	total_closes_train.extend(closes_train)
	total_closes_test.extend(closes_test)

	total_trained_closes_lin.extend(trained_closes_lin)
	total_trained_closes_pol.extend(trained_closes_pol)
	total_trained_closes_rbf.extend(trained_closes_rbf)

	total_tested_closes_lin.extend(tested_closes_lin)
	total_tested_closes_pol.extend(tested_closes_pol)
	total_tested_closes_rbf.extend(tested_closes_rbf)

print('Training...')
print('RMS Error (lin):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_lin)))
print('RMS Error (pol):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_pol)))
print('RMS Error (rbf):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_rbf)))
print('R2 Score (lin):', r2_score(total_closes_train, total_trained_closes_lin))
print('R2 Score (pol):', r2_score(total_closes_train, total_trained_closes_pol))
print('R2 Score (rbf):', r2_score(total_closes_train, total_trained_closes_rbf))

plt.plot(total_dates_train, total_closes_train, color = 'black', label = 'Actual')
plt.plot(total_dates_train, total_trained_closes_lin, color = 'red', label = 'Predicted (Linear)')
plt.plot(total_dates_train, total_trained_closes_pol, color = 'green', label = 'Predicted (Polynomial)')
plt.plot(total_dates_train, total_trained_closes_rbf, color = 'blue', label = 'Predicted (RBF)')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Stock Prediction - Kernels (Training)')
plt.legend()
plt.show()

print('Testing...')
print('RMS Error (lin):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_lin)))
print('RMS Error (pol):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_pol)))
print('RMS Error (rbf):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_rbf)))
print('R2 Score (lin):', r2_score(total_closes_test, total_tested_closes_lin))
print('R2 Score (pol):', r2_score(total_closes_test, total_tested_closes_pol))
print('R2 Score (rbf):', r2_score(total_closes_test, total_tested_closes_rbf))

plt.plot(total_dates_test, total_closes_test, color = 'black', label = 'Actual')
plt.plot(total_dates_test, total_tested_closes_lin, color = 'red', label = 'Predicted (Linear)')
plt.plot(total_dates_test, total_tested_closes_pol, color = 'green', label = 'Predicted (Polynomial)')
plt.plot(total_dates_test, total_tested_closes_rbf, color = 'blue', label = 'Predicted (RBF)')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Stock Prediction - Kernels (Testing)')
plt.legend()
plt.show()