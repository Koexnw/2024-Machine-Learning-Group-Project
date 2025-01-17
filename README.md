# 机器学习理论课大作业

我们利用Rechorus框架([https://github.com/THUwangcy/ReChorus/](https://github.com/THUwangcy/ReChorus/)) 对《Diffusion Recommender Model》一文进行复现，并参考《Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity》一文进行改进。

## 具体实现

我们把Diffrec模型继承到General Model里面，统一采用basereader，对于Diffrec和CH-Diffrec分别写了对应的runner，Diffrunner和CHDiffrunner。严格保证了评价指标的计算、数据的读取等关键步骤与其余runner一致。

## 参数设置与数据处理

**Anime**

* `w_min` = 0.3
* `w_max` = 0.7
* `steps` = 10
* `noise_scale` = 0.001
* `noise_min` = 0.0006
* `noise_max` = 0.005
* `lr` = 0.0001

**Grocery & Gourmet Food**

* `w_min` = 0.1
* `w_max` = 1.0
* `steps` = 8
* `noise_scale` = 0.003
* `noise_min` = 0.0003
* `noise_max` = 0.002
* `lr` = 0.0001

**MovieLens**

* `w_min` = 0.2
* `w_max` = 0.8
* `steps` = 8
* `noise_scale` = 0.001
* `noise_min` = 0.0005
* `noise_max` = 0.005
* `lr` = 0.0001



**数据处理**

* 使用留一法（leave-one-out）对数据集进行划分，最近的交互记录作测试集，次近的作验证集，其余作训练集。
* 在验证集和测试集中对每个测试/验证用例进行负采样，将未交互过的商品作为负样本（随机采样99个），与正样本一起进行排序，计算评价指标。
* Top-K推荐。



## Results

我们的结果如下
**Grocery & Gourmet Food:**

| Metric      | NeuMF (ReChorus) | LightGCN (ReChorus) | Diffrec (复现结果) | CH-Diffrec（改进结果）   |
|--------------|--------------------|--------------------|--------------------------|------------|
| HR@5         |  0.2977 | 0.3710             | 0.2785                    | 0.2736    |
| NDCG@5       |  0.2000 | 0.2566             | 0.1971                    | 0.1985    |
| HR@10        |  0.4073 | 0.4925             | 0.3857                    | 0.3718    |
| NDCG@10      |  0.2355 | 0.2961             | 0.2164                    | 0.2019    |


**MovieLens:**

| Metric      | NeuMF (ReChorus) | LightGCN (ReChorus) | Diffrec (复现结果) | CH-Diffrec（改进结果）   |
|--------------|--------------------|--------------------|--------------------------|------------|
| HR@5         | 0.4732              | 0.5272             | 0.5419                    | 0.5432    |
| NDCG@5       | 0.3153              | 0.3510             | 0.3567                    | 0.3543    |
| HR@10        | 0.6737              | 0.7172             | 0.7083                    | 0.7094    |
| NDCG@10      | 0.3804              | 0.4126             | 0.4172                    | 0.4105    |


**Anime**

| Metric      | NeuMF (ReChorus) | LightGCN (ReChorus) | Diffrec (复现结果) | CH-Diffrec（改进结果）   |
|--------------|--------------------|--------------------|--------------------------|------------|
| HR@5         | 0.4887              | 0.5189             | 0.5201                    | 0.5213    |
| NDCG@5       | 0.3357              | 0.3593             | 0.3604                    | 0.3597    |
| HR@10        | 0.6649              | 0.6945             | 0.6848                    | 0.6855    |
| NDCG@10      | 0.3926              | 0.4161             | 0.4191                    | 0.4197    |


## 运行我们的代码
**Diffrec参考示例**
* `python main.py --model_name Diffrec  --dataset MovieLens_1M --path ../data` 

**CH-Diffrec**

首先获取编码文件

* `python data_preprocessing.py` 

再跑模型

* `python main.py --model_name CH_Diffrec  --dataset MovieLens_1M --path ../data --n_hop=2 --pre_name=2_hop_ML_1M` 
