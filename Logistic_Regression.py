# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:18:01 2019

@author: lisssse14
"""
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
"""讀取資料集"""
bc = datasets.load_breast_cancer()
x = bc.data
y = bc.target
"""0-1正規化"""
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x)
"""80-20分割為訓練 測試"""
x_train,x_test,y_train,y_test = train_test_split(x_train_minmax,y,test_size=0.2, shuffle = True,random_state=555)
class LogisticRegression(object):
    def __init__(self, lr=0.05, n_iter=100, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        """權重初始化"""
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            self.cost_.append(self.update_weights(x,y))
            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
        return self
    def update_weights(self,x,y):
        net_input = self.net_input(x)
        output = self.sigmoid(net_input)
        errors = (y - output)
        """x轉置是為了矩陣乘法"""
        self.w_[1:] += self.lr * x.T.dot(errors)
        self.w_[0] += self.lr * errors.sum()
        cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
        return cost
        
    def net_input(self, X):
        """結果初始化"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid(self, z):
        """激發函數Sigmoid"""
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        
lrgd = LogisticRegression(lr=0.11, n_iter=1000, random_state=123)
lrgd.fit(x_train,y_train)
y_pred = lrgd.predict(x_test)
accuracy_score(y_test, y_pred)