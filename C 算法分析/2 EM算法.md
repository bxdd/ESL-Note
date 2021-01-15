# 2 EM算法

## 1 预备知识

1. 极大似然估计
2. Jensen不等式

* 如果$f$是凸函数，$X$是随机变量，那么
  $$
  E(f(X))\ge f(E[X])
  $$
  证明：
  $$
  E(f(X))=\int f(x)p(x)dx\ge f(\int xp(x)dx)=f(E(x))[琴生不等式]
  $$
  凹函数则反过来
  $$
  E(f(X))\le f(E[X])
  $$
  

## 2 EM算法详述

### 2.1 问题描述

1. 我们目前有100个男生和100个女生的身高，但是我们不知道这200个数据中哪个是男生的身高，哪个是女生的身高，即抽取得到的每个样本都不知道是从哪个分布中抽取的。这个时候，对于每个样本，就有两个未知量需要估计：
   * 这个身高数据是来自于男生数据集合还是来自于女生？
   * 男生、女生身高数据集的正态分布的参数分别是多少？

2. 基本步骤：

* 初始化参数：先初始化男生身高的正态分布的参数
* 计算每一个人更可能属于男生分布或者女生分布
* 通过分为男生的n个人来重新估计男生身高分布的参数（最大似然估计），女生分布也按照相同的方式估计出来，更新分布。
* 重复上面三步，直到参数不发生变换

### 2.2 EM算法推导

1. 对于$n$个样本观察数据$x=(x_1, x_2,...,x_n)$, 找出样本的模型参数$\theta$, 极大化模型分布的对数似然函数：
   $$
   \hat \theta = argmax\sum_{i=1}^n \log{p(x_i;\theta)}
   $$
   如果我们得到的观察数据有未观察到的隐含数据$z=(z_1,z_2,...,z_n)$ ，即上文中每个样本属于哪个分布是未知的。
   $$
   \hat \theta = argmax\sum_{i=1}^n \log{p(x_i;\theta)}=argmax\sum_{i=1}^n \log{\sum_{z_i}p(x_i,z_i;\theta)}
   $$
   引入新的分布$Q_i(z_i)$, 表示$z_i$的分布概率
   $$
   \sum_{i=1}^n \log{\sum_{z_i}p(x_i,z_i;\theta)}=\sum_{i=1}^n \log{\sum_{z_i}Q_i(z_i)\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}}
   \\ \ge \sum_{i=1}^n\sum_{z_i}Q_i(z_i)\log{\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}}
   $$
   这里可以看作对$l(\theta)$求了下界。如果$\theta$以及给定，则$l(\theta)$的值就决定于$Q_i(z_i)，p(x_i,z_i)$。 我们可以通过调整这两个概率使下界不断上升，以逼近$l(\theta)$的真实值。

   根据Jensen不等式，要想让等式成立，需要让随机变量变成常数值
   $$
   \frac{p(x_i,z_i;\theta)}{Q_i(z_i)} = c\\
   \sum_{z}Q_i(z_i) = 1 \rightarrow \sum_{z}p(x_i,z_i;\theta) = c\\
   \rightarrow Q_i(z_i) = \frac{p(x_i,z_i;\theta)}{p(x_i;\theta)} 
   =p(z_i|x_i;\theta)
   $$
   也就是固定参数后，$Q_i(z_i)$就是后验概率，这就是E步。

   则需要极大化
   $$
   argmax\sum_{i=1}^n\sum_{z_i}Q_i(z_i)\log{\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}} = \\ \sum_{i=1}^n\sum_{z_i}Q_i(z_i)\log{p(x_i,z_i;\theta)} -Q_i(z_i)\log{Q_i(z_i)}
   \\ \rightarrow argmax \sum_{i=1}^n\sum_{z_i}Q_i(z_i)\log{p(x_i,z_i;\theta)}
   $$
   这就是M步。

### 2.3 算法流程

1. 输入：观察数据$x = (x_1,x_2,...,x_n)$, 联合分布$p(x,z;\theta)$, 条件分布$p(z|x;\theta)$, 最大迭代次数$J$
2. 流程：

* 随机初始化模型参数θ的初值$\theta_0$

* $j=1,2,...,J $开始$EM$算法迭代

  * E步：计算联合分布的条件概率期望
    $$
    Q_i(z_i)=p(z_i|x_i,\theta_j)\\
    l(\theta, \theta_j) = \sum_{i=1}^n\sum_{z_i}Q_i(z_i)\log\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}
    $$

  * M步：极大化$l(\theta, \theta_j)$, 得到$\theta_{j+1}=argmax_\theta l(\theta,\theta_j)$， 如果已经收敛，则算法结束。

3. 输出参数

### 2.4 GMM