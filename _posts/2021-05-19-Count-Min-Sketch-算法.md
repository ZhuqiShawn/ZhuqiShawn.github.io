---
title: Count-min Sketch 算法
author: Zhuqi Xiao
date: 2021-05-19 12:00:00 +0800
categories: [Algorithm]
tags: [algorithm, big data]
math: true
mermaid: true
image:
  path: /assets/img/miscellaneous/nor_3.jpg
  width: 800   # in pixels
  height: 500   # in pixels
  alt: Picture shot in Luleå, Sweden
---
## 简介
Count-min Sketch算法是一个可以用来计数的算法，在数据大小非常大时，一种高效的计数算法，通过牺牲准确性提高的效率。
+ 是一个概率数据机制
+ 算法效率高
+ 提供计数上限
  

其中，重要参数包括
+ Hash 哈希函数数量： k 
+ 计数表格列的数量： m  
+ 内存中用空间： $k \times m \times \text{size of counter}$

### 举个例子🌰
我们规定一个 $m = 5$, $k = 3$ 的Count-min Sketch，用来计数，其中所有hash函数如下

$$
k\left\{ \begin{array}{lr}
    h_1(x)=\text{ASCII}(x)\\ 
    h_2(x) = 2 + \text{ASCII}(x)\\ 
    h_3(x) = 4 \cdot \text{ASCII}(x) 
    \end{array} 
    \right.$$

注意，所有hash函数的结果需 $\mod m$
下面开始填表，首先初始状态为

<center><table><tr><td></td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>$h_1$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>$h_2$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>$h_3$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table></center>

首先，向里面添加字母**B**，其ASCII码为66，求hash函数的结果为

$$\left\{ \begin{array}{lr}             
h_1(x) = 1\\ 
h_2(x) = 3\\ 
h_3(x) = 4 
\end{array} 
\right. $$

因此，表格变为

<center><table><tr><td></td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>$h_1$</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>$h_2$</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td></tr><tr><td>$h_3$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr></table></center>

接下来，我们查询字母**A**，其ASCII码为65，求hash函数的结果为

$$ \left\{ \begin{array}{lr}
    h_1(x) = 0\\ 
    h_2(x) = 2\\ 
    h_3(x) = 0 
\end{array} 
\right.$$

用这个结果去读表，发现其对应位置均为0，因此字母A最多出现0次，这个值是准确的。
然后，我们在查询字母G，其ASCII码为71，求hash函数的结果为

$$\left\{ \begin{array}{lr}
    h_1(x) = 1\\ 
    h_2(x) = 3\\ 
    h_3(x) = 4 
\end{array} 
\right. $$

用这个结果去读表，发现其对应位置均为1，因此字母G最多出现1次；出错了！我们从未向里面添加过字母G，这就是一次collision。Count-min Sketch的确会有这种问题，因为这个模型是从Bloom Filter衍生过来的。所以说Count-min Sketch是一个概率模型，返回的结果是一个上限值（upper-bound）。

## 设计最优 Count-min Sketch

有了上面的问题，我们自然而然就会想到如何设计一个最优的Count-min Sketch模型。
首先，规定一些参数：
+ 数据流大小： $n$
+ 元素 x 的真实计数值： $c_x$ 
+ 元素 x 的估计计数值： $\hat{c}_x$ 
+ 我们可以自己选择的参数： $k$ （hash函数数量）和 $m$ （表格列的数量）

注：如果我们的模型 $k = 1$, $m = n$ ，另唯一的 $h_1(x) = x$，那么我们可以得到准确的计数结果。

现在，我们希望设定一个错误范围 $(c_x\leq\hat{c}_x \leq c_x + \varepsilon n)$，这个范围表示估计值的取值范围，我们希望结果在这个范围的概率为

$$P\left(c_x\leq\hat{c}_x \leq c_x + \varepsilon n\right)\geq 1 - \delta $$

这里， $(1 - \delta)$ 表示结果在这个范围里的概率。

那么设计一个最优Count-min Sketch模型的过程为：
+ 估计数据流大小 $n$ 的大小
+ 选择一个合理的 $\varepsilon$ 值使 $\hat{c}_x - c_x \leq \varepsilon n$ 
+ 选择一个合理的概率值 $(1-\delta)$
+ $m$ 和 $k$ 的最优值可以通过以下公式获得：
  
  $$m = \left\lceil{\dfrac{e}{\varepsilon}}\right\rceil , k = \left\lceil{\ln (\dfrac{1}{\delta})}\right\rceil $$

可以看出，想要错误范围越小，就要更大的 $m$ ，也就是表格的列数；
同理，想要更高的概率（更小的 $\delta$  ），就要更大的 $k$ ，也就是更多的hash函数。

### 举个例子🌰

假设我们现在需要为大小为 $10^6$ 的数据计数，我们选择 $\varepsilon n = 2000$ ，即 $\varepsilon = 0.002$。由此我们可以得出$m = 1360$，假如我们希望 $99\%$ 的概率落在这个范围内，可得 $\delta = 0.01$，因此hash函数数量 $k = 5$
假设每个计数单元占内存大小为4 byte，那么，该模型将占用内存

$$m \times k \times\text{size of counter} = 1360 \times 5 \times 4 \text{ bytes} \approx 28\text{ kB}  $$

参考：
[Advanced Data Structures: Count-Min Sketches](https://www.youtube.com/watch?v=mPxslXpg8wA)