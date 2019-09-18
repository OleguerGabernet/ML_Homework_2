import pandas as pd
import math
import numpy as np
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

#START
#Importing stock data from YAHOO FINANCE
#=======================================

start = datetime.datetime(2010, 5, 1)    #Starting date of the stock prices file
end = datetime.datetime(2019, 9, 1)     #Ending date of the stock prices file

df = web.DataReader("AAPL", 'yahoo', start, end) #Read the stock data from a certain company and database.
df.tail()   # Show the last rows of the data. Print it to see

#Rolling Mean for the last 100 days
#===================================

close_px = df['Adj Close'] # List of all the closing prices
mavg = close_px.rolling(window=100).mean()  # Rolling Mean (or Moving Average)

#Plotting the Rolling Mean vs Raw data (closing price)
#=====================================================

# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style

#Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize = (8, 7))
mpl.__version__

#Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
plt.show()


# Expected return
#================

rets = close_px / close_px.shift(1) - 1
rets.plot(label = 'return')
mavg_rets = rets.rolling(window=50).mean()
mavg_rets.plot(label = 'mavg return')
plt.show()

# Feature engineering
#====================
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HT_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

dfreg.tail() #print it to see

#Model generation
#=================

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.pipeline import make_pipeline


# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

#Separation of training and testing of model by cross calidation train test split

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2, random_state=0)

# LINEAR AND POLYNOMIAL REGRESSION MODELS
#==========================================

# Linear regressionon 
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)


# KNN Regression
# ==================

clfknn = KNeighborsRegressor ( n_neighbors=2 )
clfknn.fit(X_train,y_train)

# EVALUATION
#===========

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

print('The linear regression confidence is ', confidencereg)
print('The quadratic regression 2 confidence is ', confidencepoly2)
print('The quadratic regression 3 confidence is ', confidencepoly3)
print('The knn regression confidence is ', confidenceknn)

last_date = dfreg.iloc[-1].name

dfreg['Linear'] = np.nan
dfreg['Poly2'] = np.nan
dfreg['Poly3'] = np.nan
dfreg['Knn'] = np.nan

lineardf = clfreg.predict(X_lately)
poly2df = clfpoly2.predict(X_lately)
poly3df = clfpoly3.predict(X_lately)
fdf = clfknn.predict(X_lately)

# Plotting the Prediction
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i, j in enumerate(lineardf):
	next_date = next_unix
	next_unix += datetime.timedelta(days=1)
	dfreg.loc[next_date] = [np.nan for j in range(len(dfreg.columns) - 4)] + [lineardf[i]] + [poly2df[i]] + [poly3df[i]] + [fdf[i]]

dfreg['Adj Close'].tail(200).plot()
dfreg['Linear'].tail(200).plot()
dfreg['Poly2'].tail(200).plot()
dfreg['Poly3'].tail(200).plot()
dfreg['Knn'].tail(200).plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
