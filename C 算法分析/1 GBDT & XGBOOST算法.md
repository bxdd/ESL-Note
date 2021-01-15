# GBDT & XGBOOST 算法

## 1 梯度提升

1. 梯度提升（Gradient boosting）是一种用于回归、分类和排序任务的机器学习技术，属于Boosting算法族的一部分。Boosting是一族可将弱学习器提升为强学习器的算法，属于集成学习（ensemble learning）的范畴。
2. Boosting：Boosting方法基于这样一种思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合所得出的判断，要比其中任何一个专家单独的判断要好。通俗地说，就是“三个臭皮匠顶个诸葛亮”的道理。

3. 梯度提升方法在迭代的每一步构建一个能够沿着梯度最陡的方向降低损（steepest-descent）的学习器来弥补已有模型的不足。

## 2 加法模型

1. GBDT算法可以看成是由K棵树组成的加法模型
   $$
   \hat y_i =\sum_{k=1}^K f_k(x_i), f_k\in F
   $$
   其中F是所有树组成的函数空间

2. 目标函数

   定义目标函数
   $$
   Obj =\sum_{i=1}^nl(y_i,\hat y_i)+\sum_{k=1}^K\Omega(f_k)
   $$
   其中$\Omega​$代表了模型的复杂度，若基模型是树模型，则树的深度、叶子节点数等指标可以反应树的复杂程度。

   Boosting采用的是前向优化算法，即从前往后，逐渐建立基模型来优化逼近目标函数。
   $$
   \hat y_i^0 = 0\\
   \hat y_i^1 = f_1(x_i) = \hat y_i^0 + f_1(x_i)\\
   \hat y_i^2 = f_1(x_i) + f_2(x_i) = \hat y_i^1 + f_2(x_i)\\
   ...\\
   \hat y_i^t = \sum_{k=1}^tf_k(x_i) = \hat y_i^{t-1} + f_2(x_i)\\
   $$
   以第t步的模型拟合为例，模型对第i个样本$x_i$的预测为：
   $$
   \hat y_i^t = \hat y_i^{t-1}+f_t(x_i), f(x_i)就是新加入的模型
   $$
   目标函数变为
   $$
   Obj^t =\sum_{i=1}^nl(y_i,\hat y_i^{t-1}+f_t(x_i))+\Omega(f_t) + constant
   $$

3. 泰勒展开

* 
  $$
  f(x+\Delta x ) = f(x)+f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2
  $$
  则
  $$
  Obj^t =\sum_{i=1}^n(l(y_i,\hat y_i^{t-1})+g_if_t(x_i) + \frac{1}{2}h_if_t^2(x))+\Omega(f_t) + constant
  $$
  如果损失函数是平方损失函数
  $$
  l(y_i,\hat y_i^{t-1}+f_t(x_i)) = (y_i-(\hat y_i^{t-1} + f_t(x_i)))^2\\
  g_i = 2(\hat y_i^{t-1} -y_i)\\
  h_i = 2
  $$
  由于$l(y_i,\hat y_i^{t-1})$已知，是一个常数，其对函数优化不会产生影响，目标函数可以写为
  $$
  Obj^t =\sum_{i=1}^n(g_if_t(x_i) + \frac{1}{2}h_if_t^2(x))+\Omega(f_t)
  $$
  只要求出每一步损失函数的一阶和二阶导的值(前一步$\hat y^{t-1}$已知，所以是常数)， 然后最优化目标函数，就可以得到每一步的$f(x)$, 最后根据加法模型得到一个整体模型。

  

## 3 GBDT算法

1. 决策树来表示上一步的目标函数

* 一颗生成好的决策树，即结构确定，叶子节点$T$片，表示为向量$w\in R^T$。则存在映射$q:R^d\rightarrow \{1,2,...,T\}$,  因此决策树可以定义为$f_t(x) = w_{q(x)}​$

* 决策树的正则项可以表达为
  $$
  \Omega(f_t) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2
  $$
  

  即决策树模型的复杂度由生成的树的叶子节点数量和叶子节点对应的值向量的L2范数决定。

* 设$I_j=\{i|q(x_i)=j\}$, 也就是第$j$个叶子节点的样本集合，则目标函数可以表示为
  $$
  Obj^{(t)} =\sum_{i=1}^n(g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i))+\Omega(f_t)
  \\ = \sum_{i=1}^n(g_iw_{q(x_i)} + \frac{1}{2}h_iw_{q(x_i)}^2)+\gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2
  \\ = \sum_{j=1}^T((\sum_{i\in I_j}g_i)w_j + \frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w_j^2)+\gamma T 
  \\ = \sum_{j=1}^T(G_jw_{j} + \frac{1}{2}(H_j+\lambda)w_{j}^2)+\gamma T
  $$
  如果树的结构是确定的，也就是$q$确定，已经知道了每个叶子结点有哪些样本。所以$G_i$和$H_i$确定，但是$w$不确定，令目标函数一阶导数为0，可以得到叶子节点$j$对应的值：
  $$
  w_j^* = -\frac{G_j}{H_j+\lambda}
  $$
  

  目标函数化简为：
  $$
  Obj = -\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda} + \gamma T
  $$

* 对于只考虑一阶导数的GBDT
  $$
  w^*_j = -\frac{G_j}{\lambda}\\
  Obj^{t} = -\sum_{j=1}^T\frac{G_j^2}{\lambda} + \gamma T
  $$
  

## 4 优化目标函数

1. 暴力

* 枚举所有结构，也就是$q$
* 计算每种树结构下的目标函数值
* 取目标函数最小（大）值为最佳的数结构, 然后再求出$w^*​$的值

2. 贪心策略

* 步骤：
  * 从深度为0的树开始，对每个叶节点枚举所有的可用特征
  * 针对每个特征，把属于该节点的训练样本根据该特征值升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的最大收益（采用最佳分裂点时的收益）
  * 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，把该节点生长出左右两个新的叶节点，并为每个新节点关联对应的样本集
  * 直到以下条件停止
    * 当引入的分裂带来的增益小于设定阀值的时候，我们可以忽略掉这个分裂，也就是$\gamma$
    * 当树达到最大深度时则停止建立决策树

* 收益计算：

  只需要紧扣目标函数，原视目标函数
  $$
  考虑二阶导数:\\
  Obj_{old} = -\frac{1}{2}[\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] + \gamma\\
  Obj_{new} = -\frac{1}{2}[\frac{(G_L)^2}{H_L+\lambda}+\frac{(G_R)^2}{H_R+\lambda}] + 2\gamma\\
  Gain = Obj_{old}-Obj_{new}
  \\ = \frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] -\gamma\\
  考虑一阶导数：\\
  Gain  = [\frac{G_L^2}{\lambda}+\frac{G_R^2}{\lambda} - \frac{(G_L+G_R)^2}{\lambda}] -\gamma
  $$

* 求出$w^*​$
* $\hat y_i^t = \hat y_i^{t-1} + \epsilon f_t(x)$, 其中$\epsilon$ 是学习率，主要是为了抑制模型过拟合。

