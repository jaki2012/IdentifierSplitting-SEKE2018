import pandas as pd
import numpy as np
import time


# 统计单词字母形态的应用 
# 大小写个数
df = pd.read_csv("tmp/non_hardsplit_bt11_oracle_samples.csv", header=None, keep_default_na=False)

values = df.values[:, :]

def find_split_positions(seqs_list):
	positions = []
	for i in range(len(seqs_list) - 1):
		if seqs_list[i] == 'S' and seqs_list[i+1] == 'S':
			positions.append(i+1)
		elif seqs_list[i] == 'S' and seqs_list[i+1] == 'B':
			positions.append(i+1)
		elif seqs_list[i] == 'E' and seqs_list[i+1] == 'B':
			positions.append(i+1)
		elif seqs_list[i] == 'E' and seqs_list[i+1] == 'S':
			positions.append(i+1)
	if len(positions) != len(set(positions)):
		print("damn it !")
	# return set(positions)
	# 要特么按顺序排
	return positions

def get_original_parts(identifier,positions):
	prev_os = 0
	parts = []
	for position in positions:
		parts.append(identifier[prev_os:position])
		prev_os = position

	parts.append(identifier[prev_os:])
	return parts

allparts = []
allparts_set = set()
for i in range(len(values)):
	identifier = values[i][0]
	parts = get_original_parts(identifier, find_split_positions(values[i][32:62]))
	for part in parts:
		if part == '_':
			continue
		allparts.append(part)
		allparts_set.add(part)

print(len(allparts))
print(len(allparts_set))
allparts_cal = []

print(np.array(allparts_cal).shape)
word_display = [0] * len(allparts_set)

print(np.array(word_display).shape)

k = 0
for word in allparts_set:
	word_display[k] = word
	a = allparts.count(word.lower())
	b = allparts.count(word.upper())
	c = allparts.count(word.capitalize())
	d = allparts.count(word)
	allparts_cal.append([a,b,c,d, (a+b+c+d)])
	k = k + 1
	if(k == 100):
		print("bre")
		break
	print("进度: ======={0}%".format(round((k) * 100 / len(allparts_set))), end="\r")
	time.sleep(0.01)

j = 0

for word in allparts_set:
	if j==100:
		break
	print(word_display[j], allparts_cal[j])

	j = j + 1
