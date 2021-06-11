# modeling-data-imbalance-with-different-losses
compare the performance of cross entropy, focal loss, and dice loss in solving the problem of data imbalance

**本项目主要试验对比focal loss, dice loss, cross entropy 在处理样本不平衡性问题上的效果。试验数据为一中文数据集，label的类目数量为20，编码用简单的CNN模型。**

# About loss
对于一训练集合$D=\{X, Y\}$,  $x_i$为其中一个样本，对应的真实值为$y_i=[y_{i0}, y_{i1}]$，$p_i=[p_{i0}, p_{i1}]$为两个类别的预测概率，其中$y_{i0}, y_{i1} \in $**{0,1}**,  $p_{i0},p_{i1} \in [0,1]$，前者为取值为0或1，后者取值范围为[0,1]。
