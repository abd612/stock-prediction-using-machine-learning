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

# No normalization

total_dates_train = []
total_dates_test = []
total_closes_train = []
total_closes_test = []

total_trained_closes_rbf = []
total_tested_closes_rbf = []

df = pd.read_csv(FILE)

dates = df['Index'].tolist()
dates = np.reshape(dates, (len(dates), 1))

closes = df['Adj Close'].tolist()
closes = np.reshape(closes, (len(closes), 1))
closes = closes.ravel()

SIZE = len(dates)

for i in range(int((1-TEST)*WINDOW), SIZE, int(WINDOW)):

	dates_train, dates_test = dates[i-int((1-TEST)*WINDOW):i], dates[i:i+int(TEST*WINDOW)]
	closes_train, closes_test = closes[i-int((1-TEST)*WINDOW):i], closes[i:i+int(TEST*WINDOW)]	

	svr_rbf = SVR(kernel = 'rbf', C = C_VAL, gamma = Y_VAL)
	svr_rbf.fit(dates_train, closes_train)

	trained_closes_rbf = svr_rbf.predict(dates_train)
	tested_closes_rbf = svr_rbf.predict(dates_test)

	total_dates_train.extend(dates_train)
	total_dates_test.extend(dates_test)
	total_closes_train.extend(closes_train)
	total_closes_test.extend(closes_test)
	total_trained_closes_rbf.extend(trained_closes_rbf)
	total_tested_closes_rbf.extend(tested_closes_rbf)

print('Training...')
print('RMS Error (No normalization):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_rbf)))
print('R2 Score (No normalization):', r2_score(total_closes_train, total_trained_closes_rbf))

plt.subplots_adjust(hspace=0.6)
plt.subplot(211)
plt.plot(total_dates_train, total_closes_train, color = 'black', label = 'Actual')
plt.plot(total_dates_train, total_trained_closes_rbf, color = 'red', label = 'Predicted')
plt.xlabel('Date (Denormalized)')
plt.ylabel('Close (Denormalized)')
plt.title('Prediction - No normalization (Training)')
plt.legend()

print('Testing...')
print('RMS Error (No normalization):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_rbf)))
print('R2 Score (No normalization):', r2_score(total_closes_test, total_tested_closes_rbf))

plt.subplot(212)
plt.plot(total_dates_test, total_closes_test, color = 'black', label = 'Actual')
plt.plot(total_dates_test, total_tested_closes_rbf, color = 'red', label = 'Predicted')
plt.xlabel('Date (Denormalized)')
plt.ylabel('Close (Denormalized)')
plt.title('Prediction - No normalization (Testing)')
plt.legend()
plt.show()

# Standard normalization

total_dates_train = []
total_dates_test = []
total_closes_train = []
total_closes_test = []

total_trained_closes_rbf = []
total_tested_closes_rbf = []

df = pd.read_csv(FILE)
stdscaler = StandardScaler()

dates = df['Index'].tolist()
dates = np.reshape(dates, (len(dates), 1))
dates = stdscaler.fit_transform(dates)

closes = df['Adj Close'].tolist()
closes = np.reshape(closes, (len(closes), 1))
closes = stdscaler.fit_transform(closes)
closes = closes.ravel()

SIZE = len(dates)

for i in range(int((1-TEST)*WINDOW), SIZE, int(WINDOW)):

	dates_train, dates_test = dates[i-int((1-TEST)*WINDOW):i], dates[i:i+int(TEST*WINDOW)]
	closes_train, closes_test = closes[i-int((1-TEST)*WINDOW):i], closes[i:i+int(TEST*WINDOW)]	

	svr_rbf = SVR(kernel = 'rbf', C = C_VAL, gamma = Y_VAL)
	svr_rbf.fit(dates_train, closes_train)

	trained_closes_rbf = svr_rbf.predict(dates_train)
	tested_closes_rbf = svr_rbf.predict(dates_test)

	total_dates_train.extend(dates_train)
	total_dates_test.extend(dates_test)
	total_closes_train.extend(closes_train)
	total_closes_test.extend(closes_test)
	total_trained_closes_rbf.extend(trained_closes_rbf)
	total_tested_closes_rbf.extend(tested_closes_rbf)

print('Training...')
print('RMS Error (Standard normalization):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_rbf)))
print('R2 Score (Standard normalization):', r2_score(total_closes_train, total_trained_closes_rbf))

plt.subplots_adjust(hspace=0.6)
plt.subplot(211)
plt.plot(total_dates_train, total_closes_train, color = 'black', label = 'Actual')
plt.plot(total_dates_train, total_trained_closes_rbf, color = 'green', label = 'Predicted')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Prediction - Standard normalization (Training)')
plt.legend()

print('Testing...')
print('RMS Error (Standard normalization):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_rbf)))
print('R2 Score (Standard normalization):', r2_score(total_closes_test, total_tested_closes_rbf))

plt.subplot(212)
plt.plot(total_dates_test, total_closes_test, color = 'black', label = 'Actual')
plt.plot(total_dates_test, total_tested_closes_rbf, color = 'green', label = 'Predicted')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Prediction - Standard normalization (Testing)')
plt.legend()
plt.show()

# MinMax normalization

total_dates_train = []
total_dates_test = []
total_closes_train = []
total_closes_test = []

total_trained_closes_rbf = []
total_tested_closes_rbf = []

df = pd.read_csv(FILE)
mmscaler = MinMaxScaler()

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

	svr_rbf = SVR(kernel = 'rbf', C = C_VAL, gamma = Y_VAL)
	svr_rbf.fit(dates_train, closes_train)

	trained_closes_rbf = svr_rbf.predict(dates_train)
	tested_closes_rbf = svr_rbf.predict(dates_test)

	total_dates_train.extend(dates_train)
	total_dates_test.extend(dates_test)
	total_closes_train.extend(closes_train)
	total_closes_test.extend(closes_test)
	total_trained_closes_rbf.extend(trained_closes_rbf)
	total_tested_closes_rbf.extend(tested_closes_rbf)

print('Training...')
print('RMS Error (MinMax normalization):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_rbf)))
print('R2 Score (MinMax normalization):', r2_score(total_closes_train, total_trained_closes_rbf))

plt.subplots_adjust(hspace=0.6)
plt.subplot(211)
plt.plot(total_dates_train, total_closes_train, color = 'black', label = 'Actual')
plt.plot(total_dates_train, total_trained_closes_rbf, color = 'blue', label = 'Predicted')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Prediction - MinMax normalization (Training)')
plt.legend()

print('Testing...')
print('RMS Error (MinMax normalization):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_rbf)))
print('R2 Score (MinMax normalization):', r2_score(total_closes_test, total_tested_closes_rbf))

plt.subplot(212)
plt.plot(total_dates_test, total_closes_test, color = 'black', label = 'Actual')
plt.plot(total_dates_test, total_tested_closes_rbf, color = 'blue', label = 'Predicted')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Prediction - MinMax normalization (Testing)')
plt.legend()
plt.show()