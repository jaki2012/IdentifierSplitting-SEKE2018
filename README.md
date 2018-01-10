# essay_pythons


## Usage:
---
### step1. 设置对应的训练集、测试集文件、svmlink目录路径
### step2. ```python separate_colums.py``` 预处理数据集 训练模型输出结果
### step3. ```python separate_colums.py -eval``` 评估模型性能


### 其余各文件作用如下：
---
### 1 ```idsplit_train.py``` 用于将oracle训练集做预处理，将其变成序列标注所需要的tag形式


##
```
python LSTM_RNN.py --train_option mixed

python LSTM_RNN.py --train_option pure_oracle

python biLSTM_RNN.py --train_option mixed

python biLSTM_RNN.py --train_option pure_oracle

```
### Training Enviroment:
#### Trained totally for 72 hours
1. Macbook pro 2015, 2.5 GHz Intel Core i7, CPU built tensorflow 1.4 - Jaki's PC
2. Nvidia GTX GEForce 1060, Ubuntu 16.04, docker of GPU built tensorflow 1.4 - Rainlf's PC
3. Nvidia GTX 970X, CUDA 8.0, Ubuntu 14.04, cNN6.1, GPU built tensorflow 1.4 - Robbie's pC
4. Alibaba ecs.gn5-c4g1.xlarge（4核 30GB，GPU计算型 gn5）	1 * NVIDIA P100

### Sub-Project related to this project
1. TRIS（有向图、最短路径—Dijkstra算法）
2. LIDS（Perl语言 CPAN）
3. CodeNet（常用程序字典、应用领域、缩写大小写）
4. [Latex TABLE Generator](http://www.tablesgenerator.com/)：支持latextable生成，导出tgn，根据latex代码生成表格等
