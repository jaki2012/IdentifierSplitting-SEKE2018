import configparser
import pandas as pd
import itertools
import os

cf = configparser.ConfigParser()
cf.read('config.ini')
nhs_oracle_samples_file = cf.get("lynx_nhs_data", "oracle_samples_file")
nhs_oracle_samples_file = "tmp/jhotdraw_oracle_samples.csv"
identifiers_file = open("tmp/identifiers_tmp.txt", 'w')
splitted_identifiers_file = open("tmp/splitted_identifiers_tmp.txt", 'w')
lids_results_file = open("tmp/lids_results_tmp.txt", 'a')

df = pd.read_csv(nhs_oracle_samples_file, header=None, keep_default_na=False)
identifiers = list(itertools.chain.from_iterable(df.values[:, 0:1]))
lendata = len(identifiers)

splitted_identifiers = list(itertools.chain.from_iterable(df.values[:, 1:2]))

def preprocess_lidsresult(raw_result):
	parts = []
	raw_parts = raw_result.strip().split(',')
	for raw_part in raw_parts:
		if raw_part.find('(') != -1:
			begin_pos = raw_part.find('<-') + 2
			end_pos = raw_part.find(')')
			parts.append(raw_part[begin_pos:end_pos])
		else:
			parts.append(raw_part)
	return '-'.join(parts)

i = 0
lids_results = []
for identifier in identifiers:
	result = os.popen("/Users/lijiechu/Documents/Lingua-IdSplitter/bin/id-splitter " + identifier)
	# print(identifier, splitted_identifiers[i], preprocess_lidsresult(result.read()))
	lids_results.append(preprocess_lidsresult(result.read()))
	i = i + 1
	print(i)

identifiers_file.write(','.join(identifiers).lower())
splitted_identifiers_file.write(','.join(splitted_identifiers).lower())
lids_results_file.write(','.join(lids_results))

print("finish")