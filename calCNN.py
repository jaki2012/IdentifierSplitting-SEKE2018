import pandas as pd
import time

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

allparts = set()
for i in range(len(values)):
	split_identifier = values[i][1]
	parts = split_identifier.split('-')
	for part in parts:
		if part == '_':
			continue
		allparts.add(part.lower())

allparts = list(allparts)

print(len(allparts))
allparts_cal = [[0] * len(allparts)]*4


for k in range(len(allparts)):
	for i in range(len(values)):
		identifier = values[i][0]
		parts = get_original_parts(identifier, find_split_positions(values[i][32:62]))
		for part in parts:
			if part == '_':
				continue
			if part == allparts[k]:
				allparts_cal[0][k] = allparts_cal[0][k] + 1
			elif part.lower() == allparts[k]:
				allparts_cal[1][k] = allparts_cal[1][k] + 1
			elif part.upper() == allparts[k]:
				allparts_cal[2][k] = allparts_cal[2][k] + 1
			elif part.capitalize() == allparts[k]:
				allparts_cal[3][k] = allparts_cal[3][k] + 1
	print("进度: ======={0}%".format(round((k + 1) * 100 / len(allparts))), end="\r")
	time.sleep(0.01)

print(allparts_cal)