###
# This is developed using the youtube chanel, sentdex (https://www.youtube.com/user/sentdex)
# These py files are written based on the sentdex playlist, "Machine Learning with Python".
# https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

# Videos 1 : 7

###

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Regression (existing libraries)

import pandas as pd
import Quandl  # Website that stores the stock tickers and intra-day trading metrics
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
#from datetime import time
import time
import pickle

df = Quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100


df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.10*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.2)


" - This part creates the classifier - "
# classifier = LinearRegression(n_jobs=-1)
classifier = svm.SVR(kernel='poly')
# classifier.fit(X_train, y_train)
# with open('linearregression.pickle','wb') as f:  ##  Saves the classifier as a pickle file
#     pickle.dump(classifier, f)

" - This part opens the saved Pickle file (the classifier) - "
pickle_in = open('linearregression.pickle','rb')
classifier = pickle.load(pickle_in)


accuracy = classifier.score(X_test, y_test)
forecast_set = classifier.predict(X_lately)
#print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan ## Creates a dataframe with a single column with 'not a number' (nan) only

last_date = df.iloc[-1].name ## name of the last date
#last_unix = last_date.timestamp() #grabs the latest time stamp
last_unix = time.mktime(last_date.timetuple())
one_day = 86400 #number of seconds in a day
next_unix = last_unix + one_day #one day past the current day (one day in the future)

for i in forecast_set:
    # iterating all the predictions and attaching them to a data frame
    # goes one day at a time and inserts the forecasted number
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

