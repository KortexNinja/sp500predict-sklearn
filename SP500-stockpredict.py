import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates=[]
prices=[]

def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[2]) )
            prices.append(float(row[1]))
    return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    indexes = list(range(0,len(dates)))
    indexes = np.reshape(indexes,(len(indexes),1))
    #svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly= SVR(kernel='poly',C=1e3,degree=2)
    svr_rbf= SVR(kernel='rbf', C=1e3,gamma='auto')

    #svr_poly.fit(indexes,prices)
    svr_rbf.fit(indexes,prices)
    #svr_lin.fit(indexes,prices)

    plt.scatter(indexes,prices, color='black',label='data')
   # plt.plot(indexes,svr_lin.predict(indexes),color='red',label='Linear')
    #plt.plot(indexes,svr_poly.predict(indexes),color='green',label='polynomial')
    plt.plot(indexes,svr_rbf.predict(indexes),color='blue',label='RBF')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0]

get_data('GSPC-6months.csv')
predicted_price= predict_prices(dates, prices,1)
print(predicted_price)
    
