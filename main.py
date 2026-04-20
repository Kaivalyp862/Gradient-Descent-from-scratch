import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Cretaing My Own Gradient Descent Class

class GDR:
    def __init__(self,epochs, learning_rate):
        self.m = 100
        self.b = -120
        self.epochs = epochs
        self.lr = learning_rate
    def fit(self, X, y):
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b)* X.ravel())

            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m )
    def predict(self,X):
        return self.m * X.ravel() + self.b
                        

gdr = GDR(learning_rate=0.001,epochs= 50)

gdr.fit(X_train,y_train)

y_pred = gdr.predict(X_test)

y_pred = gdr.predict(X_test)

print("R2_score of GDR : ",r2_score(y_test,y_pred))

# Using Sklearn's Class
lr = SGDRegressor(loss='squared_error',max_iter = 50,learning_rate= 'constant',eta0=0.001)
lr.fit(X_train,y_train)
y_pred_2 = lr.predict(X_test)
print("R2_score of Sklearn Class : ",r2_score(y_test,y_pred_2))

