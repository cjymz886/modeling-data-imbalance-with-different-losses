# modeling-data-imbalance-with-different-losses
compare the performance of cross entropy, focal loss, and dice loss in solving the problem of data imbalance

**本项目主要试验对比focal loss, dice loss, cross entropy 在处理样本不平衡性问题上的效果。试验数据为一中文数据集，label的类目数量为20，编码用简单的CNN模型。**

# About loss
![image](https://github.com/cjymz886/modeling-data-imbalance-with-different-losses/blob/main/imgs/about_loss1.png)
![image](https://github.com/cjymz886/modeling-data-imbalance-with-different-losses/blob/main/imgs/about_loss2.png)

# About data
实验用的是一个中文数据集，包含一个train.txt与test.txt文件，对应的样本数量分别为：9804，9832。label的类目数量为20，分布为：['Art', 'Literature', 'Education', 'Philosophy', 'History', 'Space', 'Energy', 'Electronics','Communication', 'Computer','Mine','Transport','Enviornment','Agriculture','Economy','Law','Medical','Military','Politics','Sports']，数据集存在极度样本不平衡性问题。训练集中数据统计详细见下表。

| label | the number of samples | the weight of samples|
| ------| ------| ------|
| Art| 740 | 0.66|
|Literature|33|14.85|
|Education|59|8.31|
|Philosophy|44|11.14|
|History|466|1.05|
|Space|640|0.77|
|Energy|32|15.32|
|Electronics|27|18.16|
|Communication|25|19.61|
|Computer|1357|0.36|
|Mine|33|14.85|
|Transport|57|8.6|
|Enviornment|1217|0.4|
|Agriculture|1021|0.48|
|Economy|1600|0.31|
|Law|51|9.61|
|Medical|51|9.61|
|Military|74|6.62|
|Politics|1024|0.48|
|Sports|1253|0.39|

从上面统计可以看出，在训练集中，有些label的样本数量很少，最少为'Communication',只有25个样本，最多为'Economy',有1600样本，呈现样本不平衡问题。样本的权重计算，是采用sklearn中compute_class_weight的balanced计算方法。数据集可以下载，[链接](https://pan.baidu.com/s/1tuMbfb2pQ50Dx3IhwPP4Bg)，密码: 6yor

# About training
部分参数说明，见下面。'normal','focal_loss'两种类型的损失函数有带权重的变体，一共有5种损失函数，详细可见code。此外，在训练中将数据集按random_seed固定随时，并取最后1000条最为验证集。
| Hyperparameter |value | Description|
| ------| ------| ------|
|loss_type|str('normal','focal_loss','dice_loss')|normal指的正常cross_entropy|
|use_weight|bool(True,False)|代表是否要用样本权重进行损失计算|
|category_weight|float(list)|对应各个label的权重值|

**训练：** python run.py train <br>
**测试：** python run.py test <br>

# About experiment
实验对比共5中类型损失，评价的指标有accuracy,precision,recall,f1-score,其训练与测试实验结果如下：

**训练结果：**
| loss | accuracy|
| ------| ------|
|cross_entropy(normal)|0.956|
|weight_cross_entropy|0.954|
|focal_loss|0.955|
|weight_focal_loss|0.944|
|dice_loss|0944|

**测试结果：**
| loss | accuracy|precision | recall|f1-score|
| ------| ------| ------| ------| ------|
|cross_entropy(normal)|0.94|0.82|0.71|0.75|
|weight_cross_entropy|0.94|0.79|0.68|0.72
|focal_loss|0.94|0.80|0.71|0.74|
|weight_focal_loss|0.94|0.80|0.72|0.75|
|dice_loss|0.94|0.75|0.76|0.75|

结果显示，5类损失函数的accuracy值是一样的。因为本实验是分类任务，accuracy指标就具备足够的说服力。这样看来，该几类损失函数在训练效果上差距并不大，而整体来看，cross_entropy最好，不仅形式最为简单，而且最为稳定。使用带权重的方式去训练，对比来看，weight_cross_entropy在F1值上表现最差，weight_focal_loss相比focal_loss有所提升。dice_loss只是在recall指标上有明显提升，但整体没有表现很好的效果。

 **各个label的F1值测试结果：**
 | label | num|cross_entropy |weight_cross_entropy|focal_loss|weight_focal_loss|dice_loss|
| ------| ------| ------| ------| ------| ------|------|
| Art| 741 | **0.93**|**0.93**|**0.93**|0.92|**0.93**|
|Literature|**34**|0.14|0.25|0.15|**0.31**|0.16|
|Education|**61**|0.65|0.70|**0.72**|0.69|0.64|
|Philosophy|**45**|0.67|**0.69**|0.64|0.61|0.46|
|History|468|0.91|**0.92**|0.91|0.88|0.89|
|Space|642|0.96|0.95|**0.96**|0.95|0.94|
|Energy|**33**|0.44|0.29|0.47|0.39|**0.48**|
|Electronics|**28**|**0.51**|0.28|0.32|0.43|0.50|
|Communication|**27**|0.62|**0.68**|0.67|0.60|0.64|
|Computer|1358|**0.98**|**0.98**|**0.98**|**0.98**|**0.98**|
|Mine|**34**|0.68|0.30|0.59|**0.77**|0.75|
|Transport|**59**|**0.81**|0.71|0.75|0.75|0.72|
|Enviornment|1218|**0.97**|**0.97**|**0.97**|0.96|0.96|
|Agriculture|1022|**0.95**|**0.95**|**0.95**|**0.95**|**0.95**|
|Economy|1601|0.94|**0.95**|**0.95**|**0.95**|**0.95**|
|Law|**52**|0.60|0.55|0.53|0.59|**0.63**|
|Medical|**53**|0.75|0.73|0.65|0.73|**0.77**|
|Military|**76**|0.63|0.59|**0.67**|0.59|0.66|
|Politics|1026|**0.95**|**0.95**|**0.95**|0.94|**0.95**|
|Sports|1254|**0.99**|**0.99**|**0.99**|**0.99**|**0.99**|
 
 
从各个label的F1值来看，并没有那个loss表现的更好。在样本特别少的label(数量<100，有11个)中，相对来说，focal_loss，dice_loss稍微好一些，各自有3个label取得最佳。对比cross_entopy，其他损失函数地区在样本少的label上表现好些，但也不完全绝对，如"Transport"; 在样本多的label上，各个损失表现趋于稳定。

# Conclusion
**通过本次实验，个人总结有以下几点感受：**<br>
1.虽然paper展示出focal_loss,dice_loss表现的多么好，说带权重的损失更适合不平衡样本，但也要看数据集所在的环境，不同场景下，可能表现更槽糕；<br>
2.本次数据集上，虽然focal_loss，dice_loss没有比cross_entropy表现多出色，也就是说并没有有效的解决不平衡性问题，但至少证明了它们跟cross_entropy一样是有效的；<br>
3.样本不平衡也要分情况，如两个label的数量比为100000：1000与10000:100，虽都是相差100倍，但前者第二个label数量更多，更容易让模型去偏向。<br>
4.解决样本不平衡问题，最好办法还是增加训练样本；实在没办法，也可以尝试下不同的损失函数。<br>

# Reference
1.[Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)<br>
2.[Dice Loss for Data-imbalanced NLP Tasks](https://www.aclweb.org/anthology/2020.acl-main.45.pdf)<br>
3.[利用Dice Loss来解决NLP任务中样本不平衡性问题](https://zhuanlan.zhihu.com/p/373821653)<br>

