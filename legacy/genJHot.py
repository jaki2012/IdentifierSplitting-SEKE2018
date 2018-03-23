import csv
import pandas as pd

jhotdraw_file = "tmp/Jhotdraw.csv"

jhotdraw_fixed_file = "tmp/jhotdraw_fixed.csv"

lynx_file = "tmp/Lynx.csv"

lynx_fixed_file = "tmp/lynx_fixed.csv"


def write_jhotdraw():
	df = pd.read_csv(lynx_file)
	data = df.values

	csvwriter = csv.writer(open(lynx_fixed_file, 'w', newline=''))
	csvwriter.writerow(["n", "identifier", "splitted_result"])
	oracle = "tmp/Oracle_2.txt"
	lines = open(oracle).readlines()
	identifiers = []
	splitted_identifiers = []
	for line in lines:
		parts = line.strip('\n').split(':')
		identifiers.append(parts[0])
		splitted_identifiers.append('-'.join((parts[1].strip(' ')).split(' ')))


	for i in range(len(data)):
		if data[i][1].lower() in identifiers:
			index = identifiers.index(data[i][1].lower())
			# print([i+1, data[i][1], splitted_identifiers[index]])
			csvwriter.writerow([i+1, data[i][1], splitted_identifiers[index]])
			identifiers.remove(identifiers[index])
			splitted_identifiers.remove(splitted_identifiers[index])
		else:
			print([i+1, data[i][1], data[i][1].lower()])
			csvwriter.writerow([i+1, data[i][1], data[i][1].lower()])
			# csvwriter.writerow([i+1, data[i][1], "nonono"])

write_jhotdraw()

