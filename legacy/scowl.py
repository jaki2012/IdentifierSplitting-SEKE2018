# txt = open("/Users/lijiechu/Documents/未命名文件夹/SCOWL-wl/words.txt")
txt = open("/Users/lijiechu/Documents/google-10000-english/google-10000-english-no-swears.txt")
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


final_words = set(scowl_words)
# print(final_words)
for final_word in final_words:
	if(len(final_word)<3):
		continue
	really_trick_bt11.write("1 2 3 4 5 6 "+ final_word+' 7'+ '\n')


print(len(final_words))