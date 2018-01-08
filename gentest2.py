from urllib import request
import random
import datetime
import pandas as pd
import itertools
import numpy as np
import sys
from multiprocessing.dummy import Pool as ThreadPool

# 捕获全局的异常
def excepthook(type, value, trace):
	'''write the unhandle exception to log'''
	print('Unhandled Error: %s: %s'%(str(type), str(value)))
	sys.__excepthook__(type, value, trace)
sys.excepthook = excepthook
# https://bugs.python.org/review/20980/diff/11367/Lib/multiprocessing/pool.py
# 更改了源码 https://bugs.python.org/review/20980/#ps11367
# issue地址:https://bugs.python.org/issue20980
# 有空试下包装成实例类CLASS方法
# 多线程不能直接写
# 把检验accuracy precison recall的工作也使用多线程来处理 返回count 1 0 10 pool.map相加

starttime = datetime.datetime.now()

base_url = "http://splitit.cs.loyola.edu/cgi/splitit.cgi"
max_int = 9999
num_of_splitting = 1
verbose = False

df = pd.read_csv("tmp/hardsplit_binkley_oracle_samples.csv", header=None, keep_default_na=False)
identifiers = list(itertools.chain.from_iterable(df.values[:, 0:1]))
# identifiers = list(itertools.chain.from_iterable(df.values[0:20, 0:2]))
# print(identifiers)
splitted_identifiers = list(itertools.chain.from_iterable(df.values[:, 1:2]))
print(','.join(splitted_identifiers))

lendata = len(identifiers)
datas = df.values[:, 0:2]

pool = ThreadPool(10)

def split_and_check(data):
	count = 0
	precision = 0
	recall = 0
	fmeasure = 0
	rand = random.randint(0, max_int)
	# handle exception of url请求
	identifier = data[0].replace('.', '_')
	url = base_url + "?&id=" + identifier + "&lang=java&n=" + str(num_of_splitting) + "&rand=" + str(rand)
	# print("proceesing ", identifier)
	body = request.urlopen(url).read()
	# print("done with", identifier)
	print(identifier, body)
	body = body.decode("utf-8")
	wrong_split = True
	softwords = body.split('\n')
	gentest_split_result = []
	for i in range(len(softwords) - 1):
		softword = softwords[i].strip('\t1234567890')
		gentest_split_result = gentest_split_result + softword.split('_')

	splitted_identifier = data[1]
	parts = splitted_identifier.split('-')
	condition = lambda part : part not in ['.', ':', '_', '~']
	parts = [x for x in filter(condition, parts)]
	# calculate precision, recall, fmeasure
	correct_splits = set([i for i in range(len(splitted_identifier)) if splitted_identifier[i:].startswith('-')])
	predict_splits = set()
	prev_pos = 0
	for i in range(len(gentest_split_result) - 1) :
		prev_pos = identifier.find(gentest_split_result[i], prev_pos) + len(gentest_split_result[i])
		predict_splits.add(prev_pos)
	precise_splits = correct_splits & predict_splits
	if len(predict_splits) == 0 and len(correct_splits) == 0:
		precision = 1
	elif len(predict_splits) == 0 and len(correct_splits) !=0:
		precision = 0
	else:
		precision = len(precise_splits) / len(predict_splits)

	if len(correct_splits) == 0 and len(predict_splits) == 0:
		recall = 1
	elif len(correct_splits) == 0 and len(predict_splits) != 0:
		recall = 0
	else:
		recall = len(precise_splits) / len(correct_splits)
	# calculate fmeasure
	if (precision + recall) == 0:
		fmeasure = 0
	else:
		fmeasure = 2 * precision * recall / (precision + recall)
	# calculate accuracy 
	if len(parts) == len(gentest_split_result):
		difference = list(set(parts).difference(set(gentest_split_result)))
		if len(difference) == 0:
			count = 1
			wrong_split = False
	if verbose and wrong_split:
		print(parts)
		print(gentest_split_result)
	return count, precision, recall, fmeasure

counts, precisons, recalls, fmeasures =  zip(*pool.map(split_and_check, datas))
pool.close()
pool.join()

total_count = 0
total_precision = 0
total_recall = 0
total_fmeasure = 0

for count, precision, recall, fmeasure in zip(*(counts, precisons, recalls,fmeasures)):
	total_count = total_count + count
	total_precision = total_precision + precision
	total_recall = total_recall + recall
	total_fmeasure = total_fmeasure + fmeasure


avg_precision = round((total_precision/lendata),3)
avg_recall = round((total_recall/lendata),3)
avg_fmeasure = round( total_fmeasure/lendata, 3)
# 限制小数点位数为2位
print("accuracy of gentest is: %.2f" % (total_count/lendata))
print("precision of gentest is: %.2f" % avg_precision)
print("recall of gentest is: %.2f" % avg_recall)
print("fmeasure of gentest is: %.2f" % avg_fmeasure )	
endtime = datetime.datetime.now()

print((endtime - starttime).total_seconds())