from urllib import request
import random
import getopt
import datetime
import csv
import pandas as pd
import itertools
import numpy as np
import sys
import os
import threading
from multiprocessing.dummy import Pool as ThreadPool


def calculate(measure):
	measures = ["accuracy", "precision", "recall", "fmeasure"]
	try:
		measure_id = measures.index(measure.lower())
	except ValueError:
		print("wrong parameters for -c option")
		sys.exit()
	dataframe = pd.read_csv("tmp/already_calculated.csv")
	values = dataframe.values[:, 1+measure_id : 1+measure_id+1]
	values = list(itertools.chain.from_iterable(values))
	len_calculated = len(values)
	total = 0
	for i in range(len_calculated):
		total = total + values[i]
	print(round(total/len_calculated, 3))


def reset_calculated_file():
	temp = open("tmp/already_calculated.csv", 'w', newline='')
	temp.write("index,accuracy,precision,recall,fmeasure")
	temp.write("\n")
	temp.close()

try:  
	opts, args = getopt.getopt(sys.argv[1:], "hrc:o:", ["help", "output="])  
except getopt.GetoptError:
	# print help information and exit:
	print()

for o, a in opts:
	# -c calculate 计算模式 分析目前的measure
	if o in ["-c"]:
		calculate(a)
		sys.exit()
	if o in ["-r"]:
		print("reseting data...")
		reset_calculated_file()
		sys.exit()




def write_to_calculated(result):	
	if mutex.acquire(1):
		temp = open("tmp/already_calculated.csv", 'a+', newline='')
		csvwriter = csv.writer(temp)
		csvwriter.writerow(result)
		mutex.release()
		temp.close()

mutex = threading.Lock()


# 捕获全局的异常
def excepthook(type, value, trace):
	'''write the unhandle exception to log'''
	print('Unhandled Error: %s: %s'%(str(type), str(value)))
	# if str(type).find("ExceptionWithTraceback") != -1:
		# os.system('python gentest2.py')
	sys.__excepthook__(type, value, trace)
sys.excepthook = excepthook
# https://bugs.python.org/review/20980/diff/11367/Lib/multiprocessing/pool.py
# 更改了源码 https://bugs.python.org/review/20980/#ps11367
# issue地址:https://bugs.python.org/issue20980
# 有空试下包装成实例类CLASS方法
# 经常会有网络断开问题 解决之
starttime = datetime.datetime.now()

base_url = "http://splitit.cs.loyola.edu/cgi/splitit.cgi"
max_int = 9999
num_of_splitting = 1
verbose = False

df1 = pd.read_csv("tmp/already_calculated.csv")
calculated_indexes = list(itertools.chain.from_iterable(df1.values[:, 0:1]))
print(len(calculated_indexes))


df = pd.read_csv("tmp/non_hardsplit_binkley_oracle_samples.csv", header=None, keep_default_na=False)
identifiers = list(itertools.chain.from_iterable(df.values[:, 0:1]))
lendata = len(identifiers)
# identifiers = list(itertools.chain.from_iterable(df.values[0:20, 0:2]))
# print(identifiers)
splitted_identifiers = list(itertools.chain.from_iterable(df.values[:, 1:2]))
indexes = range(0, lendata)

identifiers_file = open("tmp/identifiers_tmp.txt", 'w')
identifiers_file.write(','.join(identifiers))
splitted_identifiers_file = open("tmp/splitted_identifiers_tmp.txt", 'w')
splitted_identifiers_file.write(','.join(splitted_identifiers))

identifiers_file.close()
splitted_identifiers_file.close()
print("writing finished")
sys.exit()

datas = df.values[:lendata, 0:2]
datas = np.column_stack((indexes, datas))
# print(datas)

# 预处理数据 减轻多线程压力
preprocessed_datas = []
for i in range(len(datas)):
	if i not in calculated_indexes:
		preprocessed_datas.append(datas[i])

# print(preprocessed_datas)
if len(preprocessed_datas) == 0:
	print("no more data to process, so ending the procedure..")
	sys.exit()
print("preprocess finished...")

pool = ThreadPool(10)

def split_and_check(data):
	count = 0
	precision = 0
	recall = 0
	fmeasure = 0
	# if data[0] in calculated_indexes:
	# 	return count, precision, recall, fmeasure
	
	rand = random.randint(0, max_int)
	# handle exception of url请求
	identifier = data[1].replace('.', '_')
	url = base_url + "?&id=" + identifier + "&lang=java&n=" + str(num_of_splitting) + "&rand=" + str(rand)
	print("proceesing ", identifier)
	body = request.urlopen(url).read()
	# print("done with", identifier)
	# print(identifier, body)
	body = body.decode("utf-8")
	wrong_split = True
	softwords = body.split('\n')
	gentest_split_result = []
	for i in range(len(softwords) - 1):
		softword = softwords[i].strip('\t1234567890')
		gentest_split_result = gentest_split_result + softword.split('_')

	splitted_identifier = data[2]
	parts = splitted_identifier.split('-')
	condition = lambda part : part not in ['.', ':', '_', '~']
	parts = [x for x in filter(condition, parts)]

	# print(identifier)
	# print(gentest_split_result)
	# print(parts)

	# calculate precision, recall, fmeasure
	temp_right_answer = "-".join(parts)
	cleaned_identifier = identifier.replace("_","")
	temp_offset = 0
	correct_splits = set()
	for i in range(len(temp_right_answer)):
		if temp_right_answer[i:].startswith('-'):
			correct_splits.add(i - temp_offset)
			temp_offset = temp_offset + 1
	predict_splits = set()
	prev_pos = 0
	for i in range(len(gentest_split_result) - 1) :
		prev_pos = cleaned_identifier.find(gentest_split_result[i], prev_pos) + len(gentest_split_result[i])
		predict_splits.add(prev_pos)
	precise_splits = correct_splits & predict_splits
	# print(predict_splits)
	# print(correct_splits)
	
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
	write_to_calculated([int(data[0]),count, precision, recall, fmeasure])
	return count, precision, recall, fmeasure

counts, precisons, recalls, fmeasures =  zip(*pool.map(split_and_check, preprocessed_datas))
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