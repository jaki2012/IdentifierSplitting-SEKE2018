import os
import csv
import pandas as pd

# file = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"
file = "tmp/cheat_splitting_file.csv"
csv_file = "tmp/binkley_data1.csv"

csv_result = open(csv_file, 'w', newline='')
csvwriter = csv.writer(csv_result)

for line in open(file).readlines():
	line = line.strip().replace(":", "_").replace("~","_").replace(".","_").replace("___","_").replace("__","_")
	data = line.split(',')
	identfier = data[0]
	if identfier.isdigit():
		print(identfier)
		continue
	splits = data[1]
	csvwriter.writerow(data)
	# print(identfier, splits)


