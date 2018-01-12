txt = open("/Users/lijiechu/Documents/未命名文件夹/SCOWL-wl/words.txt")

trick_bt11_txt = open("tmp/trickbt11.txt")
really_trick_bt11 = open("tmp/trick11.txt", 'w')

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
# print(scowl_words)

lines = trick_bt11_txt.readlines()
for line in lines:
	line = line.strip('\n')
	words = line.split('-')
	for word in words:
		if word!= '_' and word.lower() in scowl_words:
			final_words.append(word)


final_words = set(final_words)
# print(final_words)
for final_word in final_words:
	really_trick_bt11.write("1 2 3 4 5 6 "+ final_word+' 7'+ '\n')