#coding:utf-8  
import tensorflow as tf  
import csv
import random
import string
import pandas as pd
import numpy as np
import configparser

cf = configparser.ConfigParser()
cf.read('config.ini')
oracle_samples_file = cf.get("original_oracles", "jhotdraw_oracle_samples")
input_data_file = cf.get("original_oracles", "jhotdraw_dataset_file")


def sequence_label(str):
	if len(str) == 1:
		return ['S']
	elif len(str) == 2:
		return ['B','E']
	else:
		return ['B', (len(str)-2) * 'M','E']

# 设置newline，否则两行之间会空一行
oracle_samples = open(oracle_samples_file,'w', newline='')
writer = csv.writer(oracle_samples)

INPUT_DATA = input_data_file
# file = open(INPUT_DATA)
# raw_data = file.readlines()
df = pd.read_csv(INPUT_DATA,keep_default_na=False)
raw_data = df.values
count = 0
wait_to_shuffle =[]
# record_file = open("tmp/lynx_expand.txt", 'w')
checkss= [] 
dddsss = []
for line in range(len(raw_data)):
	count = count + 1
	data = raw_data[line]
	# print(data)
	data1 = data[1]
	checkss.append(data1)

	data2 = data[2]
	# 去除一些错误的数据集和expand
	if(data1.lower()!= ''.join(data2.split('-'))):
		print("suck it", data1, data2)
		# record_file.write(str(line+1) + " " + data1 + " " + data2)
		# record_file.write('\n')
		continue
	# 被这个例外搞死了
	if data1.find("gdk_window_set_decorations")!=-1:
		data2=data2[1:]
		print(data2)
	temp_save = data1
	# bt11还需要注意大小写分割
	data1 = data1.lower()
	lenorigin = len(data1)
	lensplit = len(data2)
	split = []
	offset = 0
	for i in range(lenorigin):
		j = i + offset
		if(j< len(data2) and data1[i] == data2[j]):
			continue
		elif len(data2) <= j:
			data2 = data2 + data1[i]
			continue
		elif data2[j] == '-' and data1[i].isalnum():
			offset = offset + 1
			continue
		elif data2[j] == '-':
			data2 = data2[0:j] + '-' + data1[i] + '-' + data2[j+1:]
			offset = offset + 2
			continue
		else :
			# print(data1[i] + data2[j])
			data2 = data2[0:j] + data1[i] + data2[j:]
			# offset = offset + 1
			continue
	# print(data1, data2)
	# if data1=="_locale":
	# 	print("hahahs")
	# 	print(string.punctuation)
	data1 = temp_save.strip(string.punctuation)
	data2 = data2.strip(string.punctuation)
	# print(("count %d" % count)  + ": " + data1 + " ======= " + data2)
	# 利用这个特性做序列标注
	# if len(data1) != len(data2) - data2.count('-'):
	# 	print(("count %d" % count) + ": " + data1 + " ======= " + data2)
	for word in data2.split('-'):
		tmp = sequence_label(word)
		split =split+ tmp

	orgdata = []
	orgdata.append(data1)
	orgdata.append(data2)
	if len(data1) > 30:
		continue
	else:
		spare = 30 - len(data1)
		# if data1=="_locale":
		# 	print(spare)
		for g in range(spare):
			# if datad=="gdk_window_set_decorations":
			# 	print("aaaa" + str(len(split)))
			data1 = data1 + ' '
			split.append('N')
		# print(list(' '.join(list(data1))))
		# writer.writerow(list(' '.join(list(data1)))) 
		g = 0
		o = 1
		label = []
		length = len(data2)-1
		while g < length:
			if data2[g+1]=='-':
				g = g + 1
				label.append('1')
			else:
				label.append('0')
			g = g + 1
			o = o + 1
		while o < 30:
			o = o +1
			label.append('0')
		# print("data1: " + temp_save)
		# print("sequl: " + ''.join(split))
		# print("====")
		if(orgdata[0] in dddsss):
			continue
		dddsss.append(orgdata[0])
		wait_to_shuffle.append(orgdata + list(''.join(list(data1))) + list(''.join(split)) )
		# writer.writerow(orgdata + list(''.join(list(data1))) + list(''.join(split)) )
print(count)
print(len(dddsss))
print(len(set(dddsss)))
random.shuffle(wait_to_shuffle)
writer.writerows(wait_to_shuffle)