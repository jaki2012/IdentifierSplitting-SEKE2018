import csv
import pandas as pd

jhotdraw_file = "tmp/Jhotdraw.csv"

jhotdraw_fixed_file = "tmp/jhotdraw_fixed.csv"

lynx_file = "tmp/Lynx.csv"

lynx_fixed_file = "tmp/lynx_fixed.csv"


def write_jhotdraw():
	df = pd.read_csv(jhotdraw_file)
	data = df.values

	csvwriter = csv.writer(open(jhotdraw_fixed_file, 'w', newline=''))
	csvwriter.writerow(["n", "identifier", "splitted_result"])
	oracle = "tmp/Oracle_1.txt"
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


def see_overall(file):
	df = pd.read_csv(file)
	data = df.values
	oracle = "tmp/Oracle_2.txt"
	all_lines = open(oracle).readlines()
	i = 0
	for line in all_lines:
		parts = line.strip('\n').split(':')
		identifier = parts[1][1:]
		print(identifier, data[i][1])
		i = i+1
	# print(len(all_lines))


	
	# print(len(data))
	# count = 0
	# for i in range(len(data)):
	# 	if(i >= len(all_lines)):
	# 		print(data[i][1], "nonono")
	# 	print(data[i][1], all_lines[i].strip('\n'))
	# 	if data[i][2] == 1:
	# 		count = count + 1

	# print(count, "/", len(data))


write_jhotdraw()

