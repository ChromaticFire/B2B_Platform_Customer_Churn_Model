# -*- coding: utf-8 -*-
# @Time    : 2021/1/18 11:08
# @Author  : ccj
# @Email   : 354840621@qq.com
# @File    : adaboost测试.py
# @Software: PyCharm
import numpy as np
import math

X = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
y = np.array([1,1,1,-1,-1,-1,1,1,1,-1])

weight = [0.1 for i in range(len(X))]

base_classfier = [[[],[],[]] for i in range(len(X)+1)]

for i in range(len(base_classfier)):
    base_classfier[i][0].append(-0.5+i)
    for j in range(len(X)):
        if j < -0.5  + i:
            base_classfier[i][1].append(-1)
            base_classfier[i][2].append(1)
        else:
            base_classfier[i][1].append(1)
            base_classfier[i][2].append(-1)
# print(base_classfier)

errors_0 = math.inf
numbers_0 = 0
for i in range(len(base_classfier)):
    error = 0
    for j in range(len(y)):
        if y[j] != base_classfier[i][1][j]:
            error = error + weight[j]
    # print(error)
    if error < errors_0:
        errors_0 = error
        numbers_0 = base_classfier[i][0][0]
# print(errors_0)

errors_1 = math.inf
numbers_1 = 0
for i in range(len(base_classfier)):
    error = 0
    for j in range(len(y)):
        if y[j] != base_classfier[i][2][j]:
            error = error + weight[j]
    # print(error)
    if error < errors_1:
        errors_1 = error
        numbers_1 = base_classfier[i][0][0]

# print(errors_1)

base = 0
if errors_0 <= errors_1:
    errors = errors_0
    number = numbers_0
    base = 1
else:
    errors = errors_1
    number = numbers_1
    base = 2

print(errors)
a_1 = 0.5*math.log((1-errors)/errors)
print(a_1) #这是第一个分类器的对应的系数
print(number)#这是弱分类器的数字区别数
print(base)#这是若分类器对应的分类结果

print(base_classfier[int(number+0.5)][2])#这是第一个若分类器的具体的结果


for i in range(len(X)):
    if base_classfier[int(number+0.5)][2][i] != y[i]:
        weight[i] = weight[i] * math.exp(a_1)

print(weight)

weight_sum = sum(weight)
weight = [weight[i]/weight_sum for i in range(len(weight))]

print(weight)

#*********************************************************************************************
print("*"*150)

errors_0 = math.inf
numbers_0 = 0
for i in range(len(base_classfier)):
    error = 0
    for j in range(len(y)):
        if y[j] != base_classfier[i][1][j]:
            error = error + weight[j]
    # print(error)
    if error < errors_0:
        errors_0 = error
        numbers_0 = base_classfier[i][0][0]
# print(errors_0)

errors_1 = math.inf
numbers_1 = 0
for i in range(len(base_classfier)):
    error = 0
    for j in range(len(y)):
        if y[j] != base_classfier[i][2][j]:
            error = error + weight[j]
    # print(error)
    if error < errors_1:
        errors_1 = error
        numbers_1 = base_classfier[i][0][0]

# print(errors_1)

base = 0
if errors_0 <= errors_1:
    errors = errors_0
    number = numbers_0
    base = 1
else:
    errors = errors_1
    number = numbers_1
    base = 2

print(errors)
a_1 = 0.5*math.log((1-errors)/errors)
print(a_1) #这是第一个分类器的对应的系数
print(number)#这是弱分类器的数字区别数
print(base)#这是若分类器对应的分类结果

print(base_classfier[int(number+0.5)][2])#这是第一个若分类器的具体的结果





