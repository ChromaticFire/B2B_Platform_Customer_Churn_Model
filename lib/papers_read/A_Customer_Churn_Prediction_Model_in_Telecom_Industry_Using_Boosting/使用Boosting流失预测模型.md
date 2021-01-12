文章题目：A Customer Churn Prediction Model in Telecom Industry Using Boosting

# II. 方法论

## A.  Boosting

###  type I: AdaBoost(Adaptive Boosting)(权重没有归一化): 



 输入为： $S = \{(x_i,t_i)\}$ , $i = \{1,2,\cdots,N\}$,  $ x_i = \{(x_{i_1},x_{i_2},\cdots,x_{i_n})\}\in X$ , $t_i\in T =  \{-1,+1\}$

我们假设我们有一个使用加权数据训练基分类器的方法，得到函数$y(\mathbf x) \in\{-1,+1\}$.

- 初始化数据加权系数$\{w_n\}$,  方法是对$n=1,\cdots,N$, 令 $w_n^{(1)} = \frac{1}{N}$.

- 对于$m=1,\cdots, M:$

   - 使用训练数据调节一个分类器 $y_m(\mathbf x)$, 调节的目标是最小化加权的误差函数

     $$J_m = \sum\limits_{n=1}^{N}w_{n}^{(m)}I(y_m(\mathbf x) \neq t_n)$$

     其中$I(y_m(\mathbf x) \neq t_n)$ 是一个示性函数， 当$y_m(\mathbf x) \neq t_n$ 时， 值为 $1$, 其他情况下为 $0$ .

  - 计算
    $$
    \epsilon_m = \frac{\Sigma_{n=1}^Nw_{n}^{(m)}I(y_m(\mathbf x) \neq t_n)}{\Sigma_{n=1}^Nw_{n}^{(m)}}
    $$

  - 然后计算
    $$
    \alpha_m = ln\{\frac{1-\epsilon_m}{\epsilon_m}\}
    $$

  -  更新数据权系数
    $$
    w_{n}^{m+1} = w_{n}^{m}\text{exp}\{\alpha_m I(y_m(\mathbf x) \neq t_n)\}
    $$

-  使用最终的模型进行预测，形式为
  $$
  Y_M(\mathbf x) = \text{sign}(\sum\limits_{m=1}^{M}\alpha_m y_m(\mathbf x))
  $$
  

### 举例1(其实分类函数也是要先求出来的，并不是直接给出的.)

已知 $x_1 = (1,1),x_2 = (3,1),x_3 = (1,3)$,  $y_1 = 1, y_2 = 1, y_3 = -1$。 

我们有两个分类函数$y_1(\mathbf x): x = 2, x<2 $ 取 $-1$. $ x>2 $ 取 $+1$. 

$y_2(\mathbf x): x + y = 3, x +y<3 $ 取 $+1$. $ x +y>3$取 $-1$. 

开始时，$w_1^{(1)} = w_2^{(1)} =w_3^{(1)} =\frac{1}{3}$.

取 $m=1$ , 计算
$$
J_1 = \sum\limits_{n=1}^{3}w_{1}^{(1)}I(y_1(\mathbf x) \neq t_n)\\
= 1\times \frac{1}{3} + 0 + 0\\
= \frac{1}{3}
$$
接下来
$$
\epsilon_1 = \frac{\frac{1}{3}}{1} = \frac{1}{3}
$$
又
$$
\alpha_1 = ln\{\frac{1-\epsilon_1}{\epsilon_1}\} = ln~2
$$
从而
$$
w_1^{(1)} = \frac{2}{3}\\

w_2^{(1)} =w_3^{(1)} =\frac{1}{3}
$$


取 $m=2$ , 计算
$$
J_2 = \sum\limits_{n=1}^{3}w_{2}^{(1)}I(y_2(\mathbf x) \neq t_n)\\
= 0+ 1\times \frac{1}{3}  + 0\\
= \frac{1}{3}
$$
接下来
$$
\epsilon_2 = \frac{\frac{1}{3}}{\frac{4}{3}} = \frac{1}{4}
$$
又
$$
\alpha_2 = ln\{\frac{1-\epsilon_2}{\epsilon_2}\} = ln~3
$$
从而
$$
w_1^{(2)} = \frac{2}{3}\\

w_2^{(2)} = 1\\

w_3^{(2)} =\frac{1}{3}
$$

-  使用最终的模型进行预测，形式为
  $$
  Y_M(\mathbf x) = \text{sign}(\sum\limits_{m=1}^{M}\alpha_m y_m(\mathbf x)) = \text{sign}(ln~2*y_1(\mathbf x) + ln~3*y_2(\mathbf x))
  $$
  

对上述三个点计算结果分别为
$$
(1,1)~sign(-ln~2 + ln~3) +1\\
(3,1)~sign(-ln~2 - ln~3) -1\\
(1,3)~sign(-ln~2 - ln~3) -1\\
$$
最后的结果并没有提升。

![adaboost例子](/Users/lan/Books/《老友记》十季全：高频词汇、中英对照剧本、重点难点笔记解析及中英双语种子/ouyeel/客户流失模型/adaboost例子.png)

###  type II: AdaBoost(Adaptive Boosting)(权重归一化): 

输入：二分类的训练数据集 $\mathcal{T} = \{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\},$ $x_i\in \mathcal{X}\subseteq \pagecolor{White}\mathbf{R}^{n},y_i\in \mathcal{Y} = \{-1,+1\}.$

输出：最终分类器 $G(x)$

1. 初始化训练数据的起始权值分布

$$
D_1 = (w_{11},\cdots,w_{1i},\cdots,w_{1N}),~~w_{1i} = \frac{1}{N}, i = 1,2,\cdots, N
$$

2. 对于 $m$ 个弱分类器 $m=1,2,\cdots, M.$

   1. 在权值 $D_m$ 下训练数据集，得到弱分类器
      $$
      G_m(x): \mathcal{X}\mapsto \{-1,+1\}
      $$

   2. 计算 $G_m$ 的训练误差
      $$
      e_m = P(G_m(x_i)\neq y_i) = \sum\limits_{i=1}^{N}I(G_m(x_i)\neq y_i)
      $$

   3. 计算 $G_m$ 的系数 
      $$
      \alpha_m = \frac{1}{2}log\frac{1-e_m}{e_m}
      $$
   
4. 更新训练数据集的权值分布
      $$
      D_{m+1} = (w_{m+1,1},\cdots,w_{m+1,i},\cdots,w_{m+1,N}),\\~~w_{m+1,i} = \frac{w_{m,i}}{Z_{m}} exp(-\alpha_my_iG_{m}(x_i))
      $$
      $Z$ 是规范化因子
      $$
      Z_m = \sum\limits_{i=1}^{N}w_{mi}exp(-\alpha_my_iG_{m}(x_i))
      $$
      
3. 构建弱分类器的线性组合,
   $$
   f(x) = \sum\limits_{m=1}^{M}\alpha_mG_m(x)
   $$
   得到最终分类器
   $$
   G(x) = sign(f(x)) = sign(\sum\limits_{m=1}^{M}\alpha_mG_m(x))
   $$
   

### 两种方法的差异

在赋予权重时有差异。

对于第一种，如果该点没有被分配错误，则它的权重会不变。
$$
w_{m+1,i} = w_{m,i}
$$
 如果分配错误，则变化为
$$
w_{m+1,i} = w_{m,i}\cdot e^{\text{ln}\frac{1-e_m}{e_m}} = w_{m,i}\cdot \frac{1-e_m}{e_m}
$$
对于第二种，如果该点没有被分配错误，其权重会变化。
$$
w_{m+1,i} = \frac{w_{m,i}\cdot e^{-\frac{1}{2}\text{ln}{\frac{1-e_m}{e_m}}} }{Z_m}= \frac{w_{m,i}\cdot {(\frac{1-e_m}{e_m})}^{-\frac{1}{2}} }{Z_m}
$$
如果分配错误，则变化为
$$
w_{m+1,i} = \frac{w_{m,i}\cdot e^{\frac{1}{2}\text{ln}{\frac{1-e_m}{e_m}}} }{Z_m}= \frac{w_{m,i}\cdot {(\frac{1-e_m}{e_m})}^{\frac{1}{2}} }{Z_m}
$$
第二种方法相当于对第一种乘以一个系数:
$$
\frac{(\frac{1-e_m}{e_m})^{-\frac{1}{2}}}{Z_m}
$$

### 举例2

https://wenku.baidu.com/view/f4921c72f5ec4afe04a1b0717fd5360cba1a8dc4.html


| 序号 |  1   |  2   |  3   | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| $x$  |  0   |  1   |  2   | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|   $y$   | 1 | 1 | 1 | -1 | -1 | -1 | 1 | 1 | 1 | -1 |

初始化
$$
D_1 = (w_{11},\cdots,w_{1i},\cdots,w_{1N})，\\
w_{1,i} = 0.1, i = 1,2,\cdots,10
$$
对 $m=1$

对原来的数据进行穷举 $k = -0.5,0.5,1.5,\cdots,9.5$
$$
G_1(x)=\left\{
\begin{aligned}
1 & ， x<k\\
-1 & ，x>k
\end{aligned}
\right.
$$
​	和
$$
G_1(x)=\left\{
\begin{aligned}
-1 & ， x<k\\
1 & ，x>k
\end{aligned}
\right.
$$
最小的误差率为 $0.3$.

对应两个弱分类器
$$
G_1(x)=\left\{
\begin{aligned}
1 & ， x<2.5\\
-1 & ，x>2.5
\end{aligned}
\right.
$$
和
$$
G_1(x)=\left\{
\begin{aligned}
1 & ， x<8.5\\
-1 & ，x>8.5
\end{aligned}
\right.
$$
a. 在权值分布为 $D_1$ 的数据集上，阈值取 $2.5$ , 分类误差率最小， 基本弱分类器为
$$
G_1(x)=\left\{
\begin{aligned}
1 & ， x<2.5\\
-1 & ，x>2.5
\end{aligned}
\right.
$$
​	b.  $G_{1}(x)$ 的误差率： $e_1 = P(G_1(x_i)\neq y_i) = 0.3$

​	c. $G_{1}(x)$  的系数：
$$
\alpha_1 = \frac{1}{2}\text{ln}\frac{1-e_1}{e_1} =  \frac{1}{2}\text{ln}\frac{7}{3}  \approx 0.4236
$$
​	d.更新训练数据的权重分布
$$
D_2 = (w_{21},\cdots,w_{2i},\cdots,w_{2N})\\
D_2 = (\frac{3}{30},\frac{3}{30},\frac{3}{30},\frac{3}{30},\frac{3}{30},\frac{3}{30},\frac{7}{30},\frac{7}{30},\frac{7}{30},\frac{3}{30})(\text{没有归一化之前})\\
0.1*\text{exp}(ln(\frac{7}{3}))\textcolor{red}{(注意此处没有使用\frac{1}{2}\text{ln}\frac{7}{3}，按理说这里是不对的。)}
\\

w_{2i} = \frac{w_{1i}}{Z_1}\text{exp}(-\alpha_1y_iG_1(x_i)),i=1,2,\cdots,10\\
D_2 = (\frac{3}{42},\frac{3}{42},\frac{3}{42},\frac{3}{42},\frac{3}{42},\frac{3}{42},\frac{7}{42},\frac{7}{42},\frac{7}{42},\frac{3}{42})\\
D_2 = (0.0715,0.0715,0.0715,0.0715,0.0715,0.0715,0.1666,0.1666,0.1666,0.0715)\\

f_1(x) = 0.4236 G_1(x)
$$
弱基本分类器$G_1(x) = \text{sign}[f_1(x)]$ 在更新的数据集上有 $3$ 个误分类点。

对 $m=2$ 

a. 在权重分布 $D_2$ 上， 阈值  $v = 8.5$ 时，分类误差率最低(此处通过穷举得到只有该函数一个的误差率最低)
$$
G_2(x)=\left\{
\begin{aligned}
1 & ， x<8.5\\
-1 & ，x>8.5
\end{aligned}
\right.
$$
b. 误差率:
$$
e_2 = \frac{9}{42} \approx 0.2143
$$
c. 计算得
$$
\alpha_2 = \frac{1}{2}\text{ln}\frac{11}{3}\approx 0.6496
$$
d. 更新权值分布
$$
D_3 = (w_{31},\cdots,w_{3i},\cdots,w_{3N})\\
w_{3i} = \frac{w_{2i}}{Z_1}\text{exp}(-\alpha_2y_iG_2(x_i)),i=1,2,\cdots,10\\
D_3 = (\frac{3}{66},\frac{3}{66},\frac{3}{66},\frac{11}{66},\frac{11}{66},\frac{11}{66},\frac{7}{66},\frac{7}{66},\frac{7}{66},\frac{3}{66})\\
D_3 = (0.0455,0.0455,0.0455,0.1667,0.1667,0.1667,0.1060,0.1060,0.1060,0.0455)\\

f_2(x) = 0.4236 G_1(x) + 0.6496G_2(x)
$$
分类器 $G_2(x) = \text{sign}[f_2(x)]$ 有三个误分类点。

对 $m=3$ 

a. 在权值分布 $D_3$ 上， 阈值 $v=5.5$ 时， 分类误差率最低（此处通过穷举可得这是唯一的分类误差最低的函数）
$$
G_3(x)=\left\{
\begin{aligned}
1 & ， x<5.5\\
-1 & ，x>5.5
\end{aligned}
\right.
$$
b. 误差率：
$$
e_3 = \frac{4}{22}\approx 0.1818
$$
c. 计算
$$
\alpha_3 = \frac{1}{2}\text{ln}\frac{9}{2}\approx 0.7520
$$
d. 更新权重分布为
$$
D_4 = (\frac{27}{216},\frac{27}{216},\frac{27}{216},\frac{22}{216},\frac{22}{216},\frac{22}{216},\frac{14}{216},\frac{14}{216},\frac{14}{216},\frac{27}{216})\\
D_4 = (0.125,0.125,0.125,0.102,0.102,0.102,0.065,0.065,0.065,0.125)\\
f_3(x) = 0.4236G_1(x) + 0.6496 G_2(x) + 0.7514G_3(x)
$$
分类器 $G_3(x) = \text{sign}[f_3(x)]$ 误分类点为 $0$.

$$
G(x) = \text{sign}[f_3(x)] = \text{sign}[0.4236G_1(x) + 0.6496 G_2(x) + 0.7514G_3(x)]
$$




### gentle AdaBoost

1. 初始化每个样本 $s_i$ 的权重， $D_1(i) = 1/N.$

2. 对于 $t=1,2,\cdots,T$:

   1. 计算权重下的每一个 $y_i$ 的最小值：
      $$
      f_t = \underset{f}{arg~\text{min}}(J_t = \sum\limits_{i=1}^{N}D_t(i)(y_i-f_{t}(x_i))^2)
      $$
      
2. 取
      $$
      D_{t+1}(i) = \frac{D_{i}(i)\text{exp}(-y_if_t(x_i))}{Z_t}
      $$
      
   
   其中 $Z_t$ 是一个正规化子， 使得 $\Sigma_iD_{t+1}(i) = 1.$
   
3. 输出分类器
   $$
   \text{sign}[F(x)] = \text{sign}[\sum\limits_{t=1}^{T}f_t(x)]
   $$



### 举例3

采用例2的数据。


| 序号 |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |  10  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| $x$  |  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |
| $y$  |  1   |  1   |  1   |  -1  |  -1  |  -1  |  1   |  1   |  1   |  -1  |

初始化
$$
D_1(i) = 0.1, i=1,\cdots,10
$$
对于 $t=1$:（默认的分类器跟例 $2$ 相同）

对于例 $2$ 可知: $f_1$ 有两种选择，我们与例 $2$  的选择一样，得到
$$
f_1(x)=\left\{
\begin{aligned}
1 & ， x<2.5\\
-1 & ，x>2.5
\end{aligned}
\right.
$$
​	计算
$$
D_2(i) = \frac{D_1(i)\text{exp}(-y_if_{1}(x_i))}{Z_1}
$$
得到
$$
D_2 = (\frac{1}{10e},\frac{1}{10e},\frac{1}{10e},\frac{1}{10e},\frac{1}{10e},\frac{1}{10e},\frac{e}{10},\frac{e}{10},\frac{e}{10},\frac{1}{10e})(\text{没有归一化之前})\\
Z_1 = \frac{7}{10e} + \frac{3e}{10} \approx 1.073\\
D_2 = (\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{1}{7e^{-2}+3},\\
\frac{1}{7e^{-2}+3},\frac{1}{7e^{-2}+3},\frac{1}{7+3e^2})\\
D_2 = (0.03429,0.03429,0.03429,0.03429,0.03429,0.03429,\\
0.25330.25330.2533,0.03429)
$$
这时第 $7,8,9$ 的权重增加。

对于 $t=2$:
$$
f_2(x)=\left\{
\begin{aligned}
1 & ， x<8.5\\
-1 & ，x>8.5
\end{aligned}
\right.
$$
计算
$$
D_3(i) = \frac{D_2(i)\text{exp}(-y_if_{1}(x_i))}{Z_2}
$$
得到
$$
D_2 = (\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{1}{7+3e^2},\frac{e}{7+3e^2},\frac{e}{7+3e^2},\frac{e}{7+3e^2},\frac{1}{7e^{-2}+3},\\
\frac{1}{7e^{-2}+3},\frac{1}{7e^{-2}+3},\frac{1}{7+3e^2})(\text{没有归一化之前})\\
$$

- $\textcolor{red}{计算还没结束}$

### 举例3计算未结束

## B. Logistic Regression

样本集　$S = \{(x_i,y_i)\}$, 其中$i = \{1,2,\cdots,N\}$, $x_i = (x_{i_1},x_{i_2},\cdots, x_{i_n}) \in X$.  $y\in Y = \{+1,-1\}$.

逻辑回归的估计$P(y=1|x_i = \hat{p})$ 的概率
$$
\hat{p} = \frac{1}{1+\text{exp}(-\sum\limits_{j=1}^{n}\beta_ix_{i_j})}
$$
其中，$\beta_ix_{i_j} = \beta_0 + \beta_1 x_{i_1} + \cdots + \beta_nx_{i_n} = \text{ln}(\frac{\hat{p}}{1-\hat{p}}) = \text{logit}(\hat{p})$ . 

sigmoid函数
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$
它的图像

![sigmoid函数图像](/Users/lan/Books/《老友记》十季全：高频词汇、中英对照剧本、重点难点笔记解析及中英双语种子/ouyeel/客户流失模型/sigmoid函数图像.png)



https://zhuanlan.zhihu.com/p/74874291

逻辑回归是一个非常经典的算法，其中也包含了非常多的细节。 Logistic Regression 虽然被称为回归，但其实际上是分类模型，并常用于而分类。 Logistic 回归的本质是： 假设数据服从这个分布，然后使用极大似然估计做参数的估计。

Logistic 分布是一种连续型的概率分布，其**分布函数**和**密度函数**分别为：
$$
F(x) = P(X<x) = \frac{1}{1+ e^{-(x-\mu)/\gamma}}\\
f(x) = F'(X<x) = \frac{e^{-(x-\mu)/\gamma}}{\gamma(1+ e^{-(x-\mu)/\gamma})^2}
$$
其中，$\mu$ 表示**位置参数**， $\gamma>0$ 为**形状参数**。我们可以看下其图像特征：

![分布函数与密度函数图像](/Users/lan/Books/《老友记》十季全：高频词汇、中英对照剧本、重点难点笔记解析及中英双语种子/ouyeel/客户流失模型/分布函数与密度函数图像.jpg)

Logistic 分布是由其位置和尺度参数定义的连续分布。 Logistic 分布的形状与正态分布的形状类似，但是Logistic 分布的尾部更长，所以我们可以使用Logistic 分布来建模比正态分布具有更长尾部和更高波峰的数据分布。 在深度学习中常用到的sigmoid函数就是Logistic 分布函数在　$\mu = 0,\mu= 1$ 的特殊形式。

Logistic 回归中，以而分类为例，对于所给数据集假设存在这样的一条直线可以将数据完成线性可分。

![线性可分](/Users/lan/Books/《老友记》十季全：高频词汇、中英对照剧本、重点难点笔记解析及中英双语种子/ouyeel/客户流失模型/线性可分.jpg)

决策边界可以表示为 $w_1x_1 + w_2 x_2 + b = 0$,  假设某个样本点 $h_w(x) = w_1(x_1)+ w_2(x_2) + b>0$ 那么可以判断它的类别为 $1$, 这个过程其实是感知机。

Logistic 回归还需要加一层，它要找到分类概率 $P(Y=1)$ 与输入向量 $x$ 的直接关系， 然后通过比较概率值来判断类别。

考虑而分类问题，给定数据集

$D = (x_1,y1),(x_2,y_2),\cdots,(x_N,y_N)\subseteq R^n, y_i\in 0,1, i = 1,2,\cdots, N$

考虑到$w^T x + b$ 取值是连续的， 因此它不能你和离散变量。可以考虑用它来拟合条件概率 $p(T=1|x)$, 因为概率的取值也是连续的。

最理想的是单位阶跃函数：
$$
p(y=1|x)=\left\{
\begin{aligned}
0 & ， z<0\\
0.5 & ，z=0\\
1,&,z>0
\end{aligned}
\right.
$$
$z = w^T x + b$.

但是这个阶跃函数不可为，对数几率函数是一个常用的替代函数：
$$
y = \frac{1}{1+e^{-(w^T x + b)}}
$$
于是有：
$$
ln(\frac{y}{1-y}) =w^T x + b
$$
我们将 $y$ 视为 $x$ 为正例的概率， 则 $1-y$ 为 $x$ 为其反例的概率。两者的比值称为几率（odds）， 指该事件发生于不发生的概率比值， 将 $y$ 视为类后验概率估计重写公式有：
$$
w^T x + b=ln(\frac{P(Y=1|X)}{1-P(Y=1|X)}) \\
P(Y=1|X) = \frac{1}{1+e^{-(w^T x + b)}}
$$
也就是说，输出 $Y=1$ 的对数几率由输入 $x$ 的线性函数表示的模型，这就是逻辑回归模型。 当  $w^T x + b$ 的值越接近正无穷， $P(Y=1|X)$ 概率值也就越接近于 $1$。 

### 代价函数

设
$$
P(Y=1|X) = p(x)\\
P(Y=0|X) = 1-p(x)
$$
似然函数：
$$
L(w) = \Pi [p(x_i)]^{y_i}[1-p(x_i)]^{1-y_i}
$$
为了更方便求解，我们对等式两边同时取对数，写成对数似然函数(正常推导)：
$$
\text{ln}~L(w) = \sum [y_i\text{ln}~p(x_i)+(1-y_i)\text{ln}~(1-p(x_i))]\\
= \sum [y_i\text{ln}\frac{p(x_i)}{1-p(x_i)} + \text{ln}~(1-p(x_i))]\\
=\sum [y_i(w^T x + b) - \text{ln}~(1 + e^{w^T x + b})]
$$
在机器学习中我们有损失函数的概念，其衡量的是模型预测错误的程度。如果取整个数据集上的平均对数似然损失，我们可以得到：
$$
J(w) = -\frac{1}{N}\text{ln}~L(w)
$$
即在逻辑回归模型中，我们最大化似然函数和最小化损失函数实际上是等价的。

### 求解

求逻辑回归的方法有很多，我们这里主要讲梯度下降和牛顿法。优化的主要目标是找到一个方向，参数超这个方向移动之后使得损失函数的值能够减小，这种方向往往由一阶偏导或者二阶偏导各种组合求得。逻辑回归的损失函数为：
$$
J(w) = -\frac{1}{n}(\sum\limits_{i=1}^{n} (y_i\text{ln}~p(x_i)+(1-y_i)\text{ln}~(1-p(x_i)))
$$


### 随即梯度下降

梯度下降是通过$J(w)$ 对$w$ 的一阶导数来找下降方向的，并且以迭代的方式来更新参数，更新方式是：
$$
g_i= \frac{\partial J(w)}{\partial w_i} = (p(x_i) - y_i)x_i\\
w_i^{k+1} = w_{i}^k - \alpha g_i
$$
其中， $k$ 为迭代次数。每次更新参数后，可以通过比较 $J(w^{k+1})-J(w^{k})$ 小于阈值或者到达最大迭代次数来停止迭代。

### 注释

取　$n=1$ , 且取 $w^{T}x+b = w_1x_1+b$: 
$$
J(w) = -( y_i\text{ln}~p(x_i)+(1-y_i)\text{ln}~(1-p(x_i)) )\\
= -(y_1(w_1x_1+b)  - \text{ln} ~(1+e^{w_1x_1+b}))
$$
则
$$
\frac{\partial J}{\partial w_1} = -(y_1 x_1 -\frac{1}{1+ e^{w_1x_1+b}}\cdot x_1e^{w_1x_1+b})\\
= -(y_1x_1 - x_1 \frac{e^{w_1x_1+b}}{1+e^{w_1x_1+b}})\\
= -x_1(y_1-p(x_1))\\
= (p(x_1)-y_1)x_1
$$


### 牛顿切线法未看

















