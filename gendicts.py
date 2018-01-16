dicts = ["dicts/abbreviations.csv", "dicts/acronyms.csv", "dicts/programming.csv"]
custom_dict = "dicts/custom_lynx.csv"

trick_jhotdraw = open("tmp/trick_jhotdraw.txt", 'w')
trick_lynx = open("tmp/trick_lynx.txt", 'w')

trick_bt11_txt = open("tmp/trickbt11.txt")

dicts.append(custom_dict)

txt = open("/Users/lijiechu/Documents/未命名文件夹/SCOWL-wl/words.txt")
final_words = []

lines = txt.readlines()
count = 0
scowl_words = []
for line in lines:
	if(line.find("'") != -1):
		continue
	count = count + 1
	scowl_words.append(line.strip('\n').lower())

print(count)

final_words = []
suck  = []
lines = trick_bt11_txt.readlines()
for line in lines:
	line = line.strip('\n')
	words = line.split('-')
	for word in words:
		if word!= '_' and word.lower() in scowl_words:
			final_words.append(word)



for _dict in dicts:
	dict_file = open(_dict	)
	for line in dict_file.readlines():
		line = line.strip('\n')
		if line.find('+')!= -1:
			print(_dict,line)
			temps = line.split(',')[1].split('+')
			for temp in temps:
				if len(temp) > 1:
					final_words.append(temp)
		else:
			final_words.append(line.split(',')[0])

final_words = set(final_words)
print(len(final_words))

for final_word in final_words:
	if final_word == "nod":
		print("suck")
		continue
	trick_lynx.write("1 2 3 4 5 6 "+ final_word+' 7'+ '\n')

