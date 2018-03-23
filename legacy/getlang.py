import pandas as pd
import csv
import itertools

ORACLE_FILE = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"

GENTEST_BT11 = "tmp/gentest_binkley.csv"

ids = []
langs = []

for line in open(ORACLE_FILE).readlines():
	ids.append(line.split(' ')[1])
	langs.append(line.split(' ')[2])


df = pd.read_csv("tmp/non_hardsplit_binkley_oracle_samples.csv", header=None, keep_default_na=False)
identifiers = list(itertools.chain.from_iterable(df.values[:, 0:1]))
splitted_identifiers = list(itertools.chain.from_iterable(df.values[:, 1:2]))

csvwriter = csv.writer(open(GENTEST_BT11, 'w+', newline=''))
i = 0
count = 0
for identifier in identifiers:
	splitted_identifier = splitted_identifiers[i]
	lang = "all"
	if identifier in ids:
		index = ids.index(identifier)
		if langs[index] == "cpp":
			lang = "cplusplus"
		else:
			lang = langs[index]

	csvwriter.writerow([identifier, splitted_identifier, lang])
	i = i + 1