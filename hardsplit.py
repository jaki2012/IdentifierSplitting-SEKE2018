import pandas as pd
import csv

BT11_ORACLE = "tmp/bt11_oracle_samples.csv"
HARDSPLIT_BT11_ORACLE = "tmp/hardsplit_bt11_oracle_samples.csv"
VERBOSE = True

df = pd.read_csv(BT11_ORACLE, header=None, keep_default_na=False)
samples = df.values

num_of_identifier = len(samples)

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

def check_uncorrect_hardsplit(softwords, softsplits):
	for i in range(len(softwords)):
		if len(softsplits[i]) >0 and softwords[i][0].lower() != softsplits[i][0]:
			return False
	return True

count = 0
for i in range(num_of_identifier):
	identifier = samples[i][0]
	splitted_identifier = samples[i][1]

	hardsplit_pos = find_hardsplit_pos(identifier)

	# print(identifier, hardsplit_pos)

	softwords = []
	softsplits = []
	softwords_chars = []
	softwords_seq = []
	prev_pos = 0
	prev_sp_pos = 0
	# 偏移值
	sp_offset = 0
	for i in hardsplit_pos:
		if identifier[i] == '_':
			softwords.append(identifier[prev_pos:i])
			softsplits.append(splitted_identifier[prev_sp_pos:i+sp_offset])
			prev_pos = i + 1
			prev_sp_pos = i + 3 + sp_offset
			sp_offset = sp_offset + 2
		else:
			softwords.append(identifier[prev_pos:i+1])
			softsplits.append(splitted_identifier[prev_sp_pos:i+1+sp_offset])
			prev_pos = i + 1
			sp_offset = sp_offset + 1
			prev_sp_pos = i + 1 + sp_offset
	# 添加最后一个单词
	softwords.append(identifier[prev_pos:len(identifier)])
	softsplits.append(splitted_identifier[prev_sp_pos:len(splitted_identifier)])

	
	if len(softwords) >1 and VERBOSE and not check_uncorrect_hardsplit(softwords, softsplits):
		print(identifier, splitted_identifier, 	softwords, softsplits)
		count = count +1 
print(count)


