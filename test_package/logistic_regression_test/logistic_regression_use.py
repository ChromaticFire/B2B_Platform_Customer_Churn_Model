# -*- coding:utf-8 -*-

# @Time    : 2021/1/12 上午9:03
# @Author  : ccj
# @Email   : 354840621@qq.com
# @File    : logistic_regression_use.py
# @Software: PyCharm

# 例子的来源网址：
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

logreg = LogisticRegression(C = 1e5)
logreg.fit(X,Y)

x_min,x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min,y_max = X[:,1].min() - .5, X[:,1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1,figsize=(4,3))
plt.pcolormesh(xx,yy,Z, cmap = plt.cm.Paired)

plt.scatter(X[:,0], X[:,1], c = Y, edgecolors= 'k', cmap = plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xticks(())
plt.yticks(())

plt.show()

