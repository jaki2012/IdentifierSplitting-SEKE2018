dicts = ["dicts/abbreviations.csv", "dicts/acronyms.csv", "dicts/programming.csv"]
custom_dict = "dicts/custom_jhotdraw.csv"

trick_jhotdraw = open("tmp/trick_jhotdraw.txt", 'w')

dicts.append(custom_dict)

words = []
for _dict in dicts:
	dict_file = open(_dict	)
	for line in dict_file.readlines():
		line = line.strip('\n')
		if line.find('+')!= -1:
			print(_dict,line)
			temps = line.split(',')[1].split('+')
			for temp in temps:
				if len(temp) > 1:
					words.append(temp)
		else:
			words.append(line.split(',')[0])

for word in words:
	trick_jhotdraw.write("1 2 3 4 5 6 "+ word+' 7'+ '\n')

