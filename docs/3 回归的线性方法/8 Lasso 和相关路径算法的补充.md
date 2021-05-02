# Lasso 和相关路径算法的补充

## 1 概述

*  LAR 算法提出，许多研究都在发展对于不同问题的正则化拟合算法，这部分是一些相关的想法和以 LAR 算法为先驱的其它路径算法

## 2 增长的向前逐渐回归

* 概念：这是 **向前逐渐回归 (forward-stagewise regression)** (参考[向前逐渐回归](./3  子集的选择]))的增长版本，称为 **增长的向前逐渐回归 (Incremental Forward Stagewise Regression)**, 这是一种类似 LAR 的算法，又称$FS_{\epsilon}$

* 过程：![1619116497611](assets/1619116497611.png)

  * 假设$X,y$都经过标准化，均值为0

  * 定义$c(\hat\mu )$为当前的相关系数
    $$
    c(\hat \mu=X\hat\beta)=X^Tr=X^T(y-X\hat\beta)
    $$

  * 选择相关系数绝对值最大的变量，并且向概变量移动一个小变量$\epsilon$
    $$
    \hat j=\arg\max_{j}(|c_j|)\\
    \beta_j = \beta_j+ \delta_j =\beta_j+\epsilon sign(c_{\hat j})\\
    r = r-\epsilon\times sign(c_{\hat j})\mathbf{x}_j
    $$

  * 若$\delta_j=\frac{<r,x_j>}{<x_j,x_J>}$， 其就是向前逐渐回归了

* 系数路径

  * 图像![1619117044818](assets/1619117044818.png)

    * 以前列腺癌为例
    * 左侧$\epsilon = 0.01$, 右侧$\epsilon \rightarrow 0$。

  * 在$\epsilon \rightarrow 0$这种情形下与 lasso 路径相同。这个极限过程为 **无穷小的向前逐渐回归 (infinitesimal forward stagewise regression)** ，或者$FS_0$

    * 其与 LAR 算法均允许每个连结变量 (tied predictor) 以一种平衡的方式更新他们的系数，并且在相关性方面保持连结
    * 但是 LAR 在这些连结预测变量中的最小二乘拟合可以导致系数向相反的方向移动到它们的相关系数，因此需要对 LAR​ 算法进行修正

  * LAR 关于 $FS_0$的修正

    * 在 LAR 算法的第 4 步(见[LAR 算法](./3 子集的选择))中，系数朝着联合最小二乘方向移动，注意此时方向与最小二乘方向可能一致或者相反。这是因为LAR这里相关性相同指的是相关系数绝对值相同，所以系数增长方向能是相反的

    * 然而 $FS_0$ 中的移动方向始终与最小二乘方向保持一致

    * 因此对 LAR 算法的第 4 步修正如下

      ![1619117962078](assets/1619117962078.png)

    * 这个修正相当于一个非负的最小二乘拟合, 保持系数的符号与相关系数的符号一致
    * 这样$FS_0$的路径也可以通过$LAR$计算出来

  * 对于 LAR 算法来说

    * 若各个系数均是单调不减或者单调不增， 则LAR, lasso, $FS_0$的路径是一致的
    * 若各个系数均不过$0$, 则LAR 的 lasso是一致的

  * $FS_0$和lasso对比

    * $FS_0$比lasso 约束更强，可以看成lasso 的单调版本，其系数曲线更光滑，所以有更小的方差（TODO: 不理解光滑和小方差）

    * $FS_0$比lasso 更加复杂

      * lasso 是$\beta$ 以$L_1$范数为方向，进行单位增正后，最优化达到的残差平方和

      * $FS_0$ 是$\beta$ 在沿着系数路径$L_1$弧长为方向，进行单位增长后，最优化达到的残差平方和。这是因为$FS_0$的系数不会轻易改变方向，所以$L_1$范数就是$L_1$弧长

        > $L_1$弧长（$L_1$ arc length）：对可到曲线$\beta(s), s\in[0,S]$的$L_1$弧长为$TV(\beta,S)=\int_{0}^S \|\frac{\partial \beta}{\partial s}\|_1ds$, 对于分段函数（LAR函数曲线来说，其系数$L_1$弧长就是各个段系数$L_1$范数变换的和。

## 3 分段线性路径算法

* 