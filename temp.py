import os
import csv

file = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"
csv_file = "tmp/binkley_data.csv"

csv_result = open(csv_file, 'w', newline='')
csvwriter = csv.writer(csv_result)

for line in open(file).readlines():
	line = line.strip().replace(".","_").replace("___","_").replace("__","_")
	data = line.split(' ')
	identfier = data[1]
	splits = data[6]
	csvwriter.writerow([identfier, splits])
	print(identfier, splits)


