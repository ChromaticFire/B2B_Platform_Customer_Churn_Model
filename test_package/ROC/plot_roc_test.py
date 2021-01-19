# -*- coding: utf-8 -*-
# @Time    : 2021/1/19 8:53
# @Author  : ccj
# @Email   : 354840621@qq.com
# @File    : plot_roc_test.py
# @Software: PyCharm

# 输入y_test 与 y_score 即可.



import matplotlib.pyplot as plt
from  sklearn.metrics import roc_curve,auc


# y_test  = [1,0,1,1,0]
# y_score = [1,1,0,1,0]

y_test  = [1,1,0,1,1,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0]
y_score = [0.9,0.8,0.7,0.6,0.55,0.54,0.53,0.52,0.51,0.505,0.4,0.39,0.38,0.37,0.36,0.35,0.34,0.33,0.30,0.1]


# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值


plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()