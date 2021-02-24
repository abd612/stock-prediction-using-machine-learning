TEST = 0.20 #0.2
WINDOW = 25 #25
C_VAL = 1e2 ##1e2
Y_VAL = 1e1 #1e1
FILE = 'aapl-1y.csv' #'aapl-1y.csv' 

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
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

total_trained_closes_svr = []
total_trained_closes_mlp = []
total_trained_closes_reg = []

total_tested_closes_svr = []
total_tested_closes_mlp = []
total_tested_closes_reg = []

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

svr = SVR(kernel = 'rbf', C = C_VAL, gamma = Y_VAL)
mlp = MLPRegressor(hidden_layer_sizes = (100))
reg = LinearRegression()

for i in range(int((1-TEST)*WINDOW), SIZE, int(WINDOW)):

	dates_train, dates_test = dates[i-int((1-TEST)*WINDOW):i], dates[i:i+int(TEST*WINDOW)]
	closes_train, closes_test = closes[i-int((1-TEST)*WINDOW):i], closes[i:i+int(TEST*WINDOW)]	

	svr.fit(dates_train, closes_train)
	mlp.fit(dates_train, closes_train)
	reg.fit(dates_train, closes_train)

	trained_closes_svr = svr.predict(dates_train)
	tested_closes_svr = svr.predict(dates_test)

	trained_closes_mlp = mlp.predict(dates_train)
	tested_closes_mlp = mlp.predict(dates_test)

	trained_closes_reg = reg.predict(dates_train)
	tested_closes_reg = reg.predict(dates_test)

	total_dates_train.extend(dates_train)
	total_dates_test.extend(dates_test)
	total_closes_train.extend(closes_train)
	total_closes_test.extend(closes_test)

	total_trained_closes_svr.extend(trained_closes_svr)
	total_tested_closes_svr.extend(tested_closes_svr)

	total_trained_closes_mlp.extend(trained_closes_mlp)
	total_tested_closes_mlp.extend(tested_closes_mlp)

	total_trained_closes_reg.extend(trained_closes_reg)
	total_tested_closes_reg.extend(tested_closes_reg)

print('Training...')
print('RMS Error (svr):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_svr)))
print('RMS Error (mlp):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_mlp)))
print('RMS Error (reg):', sqrt(mean_squared_error(total_closes_train, total_trained_closes_reg)))
print('R2 Score (svr):', r2_score(total_closes_train, total_trained_closes_svr))
print('R2 Score (mlp):', r2_score(total_closes_train, total_trained_closes_mlp))
print('R2 Score (reg):', r2_score(total_closes_train, total_trained_closes_reg))

plt.plot(total_dates_train, total_closes_train, color = 'black', label = 'Actual')
plt.plot(total_dates_train, total_trained_closes_svr, color = 'red', label = 'Predicted (svr)')
plt.plot(total_dates_train, total_trained_closes_mlp, color = 'green', label = 'Predicted (mlp)')
plt.plot(total_dates_train, total_trained_closes_reg, color = 'blue', label = 'Predicted (reg)')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Stock Prediction - Models (Training)')
plt.legend()
plt.show()

print('Testing...')
print('RMS Error (svr):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_svr)))
print('RMS Error (mlp):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_mlp)))
print('RMS Error (reg):', sqrt(mean_squared_error(total_closes_test, total_tested_closes_reg)))
print('R2 Score (svr):', r2_score(total_closes_test, total_tested_closes_svr))
print('R2 Score (mlp):', r2_score(total_closes_test, total_tested_closes_mlp))
print('R2 Score (reg):', r2_score(total_closes_test, total_tested_closes_reg))

plt.plot(total_dates_test, total_closes_test, color = 'black', label = 'Actual')
plt.plot(total_dates_test, total_tested_closes_svr, color = 'red', label = 'Predicted (svr)')
plt.plot(total_dates_test, total_tested_closes_mlp, color = 'green', label = 'Predicted (mlp)')
plt.plot(total_dates_train, total_trained_closes_reg, color = 'blue', label = 'Predicted (reg)')
plt.xlabel('Date (Normalized)')
plt.ylabel('Close (Normalized)')
plt.title('Stock Prediction - Models (Testing)')
plt.legend()
plt.show()