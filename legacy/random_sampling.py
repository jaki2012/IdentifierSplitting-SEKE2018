import pandas as pd
import csv 
import random
from numpy.random import RandomState
import numpy as np
# read_table的默认分隔符为'\t'
# 由于无法一次性将文件读取 故无法读取到文件的总行数
# 可通过设置配置文件的方式来进行处理
# reader = pd.read_table('tmp/cheat_splitting_file1.csv', sep=',', header=None, chunksize=100000, iterator = True)
# sample_csv = open('tmp/sample1.csv', 'w', newline='')
# csvwriter = csv.writer(sample_csv)
# count = 1
# for chunk in reader:
# 	print(count)
# 	df = chunk.sample(frac = 0.01)
# 	# read_table会将格式变成一行
# 	data = df.values
# 	# print(data.size)
# 	csvwriter.writerows(data)
# 	count = count + 1

# list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
# # 种子数一样 就一样
# rdm = RandomState(1)
# dataset_size = 2
# X = rdm.randint(1,10, [dataset_size,dataset_size])
# # slice = random.sample(list, 5)  #从list中随机获取5个元素，作为一个片断返回  
# # print(slice) 
# print(X)

# a = [[1,2,3],[4,5,6]]
# b = [[7,8,9]]
# print(len(a))
# print(np.row_stack([a,b]))

import tensorflow as tf    
x=tf.constant([[[1,3,2],[4,5,6],[7,8,9]]])    
  
xShape=tf.shape(x)  
z1=tf.arg_max(x,2)#沿axis=0操作  
  
  
with tf.Session() as sess:  
    xShapeValue,d1=sess.run([xShape,z1])
    print(sess.run(x))  
    print('shape= %s'%(xShapeValue))  
    print(d1)  
