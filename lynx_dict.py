import pandas as pd

txt = open("/Users/lijiechu/Documents/未命名文件夹/SCOWL-wl/words.txt")
df = pd.read_csv("tmp/lynx_oracle_samples.csv", header=None)
data = df.values

dicts = set()

lines = txt.readlines()
count = 0
scowl_words = []
for line in lines:
	if(line.find("'") != -1):
		continue
	count = count + 1
	scowl_words.append(line.strip('\n').lower())

print(len(scowl_words))


for i in range(len(data)):
	splits = data[i][1].split('-')
	for split in splits:
		if split != '_' and len(split)>1 and split in scowl_words:
			dicts.add(split)


print(len(dicts))




dictis = ["dicts/abbreviations.csv", "dicts/acronyms.csv", "dicts/programming.csv"]
custom_dict = "dicts/custom_lynx.csv"

dictis.append(custom_dict)
trick_lynx = open("tmp/trick_lynx.txt", 'w')

for _dicti in dictis:
	dict_file = open(_dicti)
	for line in dict_file.readlines():
		line = line.strip('\n')
		# 针对custom——jhotdraw
		if line.find('+')!= -1:
			temps = line.split(',')[1].split('+')
			for temp in temps:
				if len(temp) > 1:
					dicts.add(temp)
		else:
			dicts.add(line.split(',')[0])


print(len(dicts))


for _dictword in dicts:
	trick_lynx.write("1 2 3 4 5 6 "+ _dictword+' 7'+ '\n')
