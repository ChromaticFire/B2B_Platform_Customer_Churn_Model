## Z-test

Z-test也称为u-test. 是一种假设检验方法。

以及魏宗舒第二版《概率论与数理统计》$\S$ 7.2 参数假设检验 

#  参数假设检验 

## $u$ 检验

设 $\xi_1,\xi_2,\cdots,\xi_n$ 取自正态母体 $N(\mu, \sigma^2)$ 的一个子样， $\sigma^2=\sigma_0^2$ 为已知常数.  要检验假设： $H_0: \mu = \mu_0, H_1 :\mu\neq \mu_0.$

如果原假设  $H_0: \mu = \mu_0$ 为真， 那么子样均值 $\overline{\xi}$ 应当在 $\xi_0$ 周围随机地摆动， 而不会偏离 $\xi_0$ 太大. 所以临界域的结构形如
$$
(|\overline{\xi}-\mu_0| >k)
$$
为了便于查表，我们把统计量 $|\overline{\xi}-\mu_0|$ 改成

$$
u = \frac{\overline{\xi}-\mu_0}{\frac{\sigma_0}{\sqrt{n}}}
$$
在 $H_0$ 为真时， 它服从标准正态分布 $N(0,1).$

给出显著性水平 $\alpha$, 当 $H_0$ 为真时
$$
P_{H_0}(|u|\geq u_{1-\frac{\alpha}{2}} )= \alpha
$$
这里 $u_{1-\frac{\alpha}{2}}$  为正态 $N(0,1)$ 表查出的 $1- \frac{\alpha}{2}$ 分位点。这样，我们便得到临界域 $C =(|u| \geq u_{1-\frac{\alpha}{2}}) $. 再从子样观察值算出 $u$ 的值。 若 $|u|\geq u_{1-\frac{\alpha}{2}}$ ,则拒绝原假设：   $H_0: \mu = \mu_0$ ， 并认为母体均值与原假设 $\mu_0$ 有显著差异， 这种检验方法称为 $u$ 检验。

## $u$ 检验例子

**例** 设某厂一车床生产的纽扣，据经验其直径服从正态分布 $N(\mu, \sigma^2), \sigma^2 = 5.2.$ 为了检验这一车床生产是否正常，现抽取容量 $n=100$ 的子样， 其子样均值 $\overline{x} = 26.56.$ 要求在显著性水平 $\alpha  =0.05$ 下检验生产是否正常。

首先，按题意，生产正常时 $\mu = \mu_0 = 26$,  而生产不正常时 $\mu\neq \mu_0$, 因此可以设立如下假设：
$$
H_0 : \mu = \mu_0 = 26, H_1 : \mu\neq \mu_0
$$
由正态分布 $N(0,1)$ 表查得 $1-\frac{\alpha}{2} = 0.975$ 分位数 $u_{0.975} = 1.96.$  根据子样观察值算得
$$
|u| = |\frac{\overline{\xi}-\mu_0}{\frac{\sigma_0}{\sqrt{n}}}| = \frac{26.56-26}{\frac{5.2}{10}} = 1.08 < 1.96
$$
不能拒绝原假设 $H_0$,  因此认为生产是正常的。



































