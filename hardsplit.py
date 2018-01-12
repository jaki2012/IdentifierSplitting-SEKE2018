import pandas as pd
import configparser
import random
import numpy as np
import csv

cf = configparser.ConfigParser()
cf.read('config.ini')
oracle_samples_file = cf.get("original_oracles", "bt11_oracle_samples")
hs_oracle_samples_file = cf.get("bt11_hs_data", "oracle_samples_file")
nhs_oracle_samples_file = cf.get("bt11_nhs_data", "oracle_samples_file")

VERBOSE = True

df = pd.read_csv(oracle_samples_file, header=None, keep_default_na=False)
num_of_identifier = len(df.values)
random_ind = list(range(0, num_of_identifier))
# random.shuffle(random_ind)
samples = df.values[random_ind[:], :]

hardsplit_bt11_result_csv = open(hs_oracle_samples_file, 'w', newline='')
csvwriter = csv.writer(hardsplit_bt11_result_csv)

non_hardsplit_bt11_result_csv = open(nhs_oracle_samples_file, 'w', newline='')
csvwriter2 = csv.writer(non_hardsplit_bt11_result_csv)

# 查看最大长度
max_length = 30	



# TO-DO merge this two function
def padding_chars(chars_list):
	spare = max_length - len(chars_list)
	for i in range(spare):
		chars_list.append(' ')
	return chars_list

def padding_seqs(seqs_list):
	spare = max_length - len(seqs_list)
	for i in range(spare):
		seqs_list.append('N')
	return seqs_list

def generate_softsplit(softword, softword_seqs):
	softsplit_parts = []
	offset = 0
	for i in range(len(softword_seqs) - 1):
		if softword_seqs[i] == 'S' and softword_seqs[i+1] == 'S':
			softsplit_parts.append(softword[i])
			offset = i+1
		elif softword_seqs[i] == 'S' and softword_seqs[i+1] == 'B':
			softsplit_parts.append(softword[i])
			offset = i+1
		elif softword_seqs[i] == 'E' and softword_seqs[i+1] == 'B':
			softsplit_parts.append(softword[offset:i+1])
			offset = i+1
		elif softword_seqs[i] == 'E' and softword_seqs[i+1] == 'S':
			softsplit_parts.append(softword[offset:i+1])
			offset = i+1
	softsplit_parts.append(softword[offset:len(softword)])
	# return '-'.join(softsplit_parts).lower()
	return '-'.join(softsplit_parts)

def find_hardsplit_pos(identifier):
	pos = []
	for i in range(len(identifier) - 1):
		if identifier[i] == '_':
			pos.append(i)
		elif identifier[i].islower() and identifier[i+1].isupper():
			pos.append(i)
		else:
			continue
	return pos

def check_uncorrect_hardsplit(softwords_seqs):
	# 如果开头不是B或者S的话，必然是没有正确hard split的结果
	for i in range(len(softwords_seqs)):
		if softwords_seqs[i][0] not in ['B', 'S']:
			return False
	return True

count = 0
full_softword_set = []
full_softsplit_set = []
total = set()
kk = 0
uu = 0
for h in range(num_of_identifier):
	identifier = samples[h][0]
	splitted_identifier = samples[h][1]
	lw_identifier = identifier.lower()
	hardsplit_pos = find_hardsplit_pos(identifier)

	# print(identifier, hardsplit_pos)

	softwords = []
	# 基于softsplt内部的复杂结构，需要后期根据序列标注结果去除
	softsplits = []
	softwords_chars = []
	softwords_seqs = []
	prev_pos = 0


	for i in hardsplit_pos:
		if identifier[i] == '_':
			softwords.append(identifier[prev_pos:i])
			softwords_chars.append(list(samples[h][prev_pos + 2:i+2]))
			softwords_seqs.append(list(samples[h][prev_pos + 2 + 30 :i+2 +30]))
			softsplits.append(generate_softsplit(identifier[prev_pos:i], list(samples[h][prev_pos + 2 + 30 :i+2 +30])))
			prev_pos = i + 1
		else:
			softwords.append(identifier[prev_pos:i+1])
			softwords_chars.append(list(samples[h][prev_pos + 2:i+1+2]))
			softwords_seqs.append(list(samples[h][prev_pos + 2 + 30 :i+ 1 +2 +30]))
			softsplits.append(generate_softsplit(identifier[prev_pos:i+1],list(samples[h][prev_pos + 2 + 30 :i+ 1 +2 +30]) ))
			prev_pos = i + 1
	# 添加最后一个单词
	softwords.append(identifier[prev_pos:len(identifier)])
	softwords_chars.append(list(samples[h][prev_pos + 2:len(identifier)+2]))
	softwords_seqs.append(list(samples[h][prev_pos + 2 + 30 :len(identifier)+2 +30]))
	softsplits.append(generate_softsplit(identifier[prev_pos:len(identifier)], list(samples[h][prev_pos + 2 + 30 :len(identifier)+2 +30])))

	# 统计、输出、并去除错误的结果
	# print(identifier, softwords)
	if len(softwords) >1 and VERBOSE and not check_uncorrect_hardsplit(softwords_seqs):
		# print(identifier, softwords, softwords_chars, softwords_seqs)
		count = count +1
		continue

	num_of_softwords = len(softwords)
	print(h)
	for k in range(num_of_softwords):
		orgdata = []
		if(len(softwords[k]) > max_length):
			max_length = len(softwords[k])
		orgdata.append(softwords[k])
		orgdata.append(softsplits[k])
		full_softword_set.append(softwords[k])
		full_softsplit_set.append(softsplits[k])
		# # remove same samples already in the dataset
		# unique_softiterm = softwords[k] + softsplits[k] + ''.join(softwords_chars[k]) + ''.join(softwords_seqs[k])
		# if unique_softiterm in total:
		# 	continue
		# else:
		# 	total.add(softwords[k] + softsplits[k] + ''.join(softwords_chars[k]) + ''.join(softwords_seqs[k]))
		index_info = str(h) + "_" + str(k) + "_" +str(num_of_softwords)
		csvwriter.writerow(orgdata + padding_chars(softwords_chars[k]) + padding_seqs(softwords_seqs[k]) + [index_info])
		
	
	# 过滤后的softsplit
	nhs_orgdata = []
	nhs_orgdata.append(identifier)
	nhs_orgdata.append(splitted_identifier)
	csvwriter2.writerow(nhs_orgdata + list(samples[h][2:32]) + list(samples[h][32:62]))
	print(h)

# 存在重复项
print(len(set(full_softword_set)))
print(len(set(full_softsplit_set)))
print(len(set(total)))
print("uncorrect hardsplit identifiers: %d" % count)
print("max_length of these soft_identifiers is %d" % max_length)

# generate_softsplit("goodabluck", ['B','M','M','E','S','S','B','M','M','E'])

