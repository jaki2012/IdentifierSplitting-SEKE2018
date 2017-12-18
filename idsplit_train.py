#coding:utf-8  
import tensorflow as tf  
import csv
import random
import string
import numpy as np


def sequence_label(str):
	if len(str) == 1:
		return ['S']
	elif len(str) == 2:
		return ['B','E']
	else:
		return ['B', (len(str)-2) * 'M','E']

# 设置newline，否则两行之间会空一行
oracle_samples = open('oracle_samples.csv','w', newline='')
writer = csv.writer(oracle_samples)

INPUT_DATA = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"
file = open(INPUT_DATA)
raw_data = file.readlines()
count = 0
wait_to_shuffle =[]
for line in raw_data:
	count = count + 1
	data = line.split(' ')
	data1 = data[1]
	data2 = data[6]
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
	data1 = data1.strip(string.punctuation)
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
	if len(data1) > 25:
		continue
	else:
		spare = 25 - len(data1)
		for g in range (spare):
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
		while o < 25:
			o = o +1
			label.append('0')
		print("data1: " + data1)
		print("sequl: " + ''.join(split))
		print("====")
		wait_to_shuffle.append(orgdata + list(''.join(list(data1))) + list(''.join(split)) )
		# writer.writerow(orgdata + list(''.join(list(data1))) + list(''.join(split)) )
random.shuffle(wait_to_shuffle)
writer.writerows(wait_to_shuffle)