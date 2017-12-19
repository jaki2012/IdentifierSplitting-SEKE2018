# essay_pythons


## Usage:
---
### step1. 设置对应的训练集、测试集文件、svmlink目录路径
### step2. ```python separate_colums.py``` 预处理数据集 训练模型输出结果
### step3. ```python separate_colums.py -eval``` 评估模型性能


## 其余各文件作用如下：
---
### 1 ```idsplit_train.py``` 用于将oracle训练集做预处理，将其变成序列标注所需要的tag形式


##
```
python LSTM_RNN.py --train_option mixed

python LSTM_RNN.py --train_option pure_oracle

python biLSTM_RNN.py --train_option mixed

python biLSTM_RNN.py --train_option pure_oracle

```