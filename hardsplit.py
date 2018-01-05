import pandas as pd
import csv

BT11_ORACLE = "tmp/bt11_oracle_samples.csv"
HARDSPLIT_BT11_ORACLE = "tmp/hardsplit_bt11_oracle_samples.csv"

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

for i in range(num_of_identifier):
	identifier = samples[i][0]
	lw_identifier = identifier.lower()
	splitted_identifier = samples[i][1]

	hardsplit_pos = find_hardsplit_pos(identifier)

	# print(identifier, hardsplit_pos)

	softwords = []
	softsplit = []
	softwords_chars = []
	softwords_seq = []
	prev_pos = 0
	for i in hardsplit_pos:
		if identifier[i] == '_':
			softwords.append(identifier[prev_pos:i])
			prev_pos = i + 1
		else:
			softwords.append(identifier[prev_pos:i+1])
			prev_pos = i+1

	softwords.append(identifier[prev_pos:len(identifier)])


	if len(softwords) >1 and identifier.find("Order")!=-1:
		print(identifier, softwords)


