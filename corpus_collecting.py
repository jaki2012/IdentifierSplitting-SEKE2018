from enchant.tokenize import get_tokenizer
import pandas as pd
import csv

tknzr = get_tokenizer("en_US")
texts = [w for w in tknzr("thisissomesimpletext")]

ORACLE_FILE = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"
GOOGLE_NORMAL_WORDS_FILE = "/Users/lijiechu/Documents/google-10000-english/google-10000-english-no-swears.txt"
CHEAT_FILE = "tmp/cheat_file.csv"

def trick_on_dataset():
	df = pd.read_csv(CHEAT_FILE, header=None, keep_default_na=False)
	data = df.values
	# 这样裁剪出来的依然是二维的
	cheat_words = data[:, 1:]
	num_of_cheat_words = len(cheat_words)
	# print(cheat_words)
	# 加入大小写，首字母大写等形态
	modified_words = []
	for i in range(num_of_cheat_words):
		cheat_word = cheat_words[i][0]
		modified_words.append(cheat_word.lower())
		modified_words.append(cheat_word.upper())
		modified_words.append(cheat_word.capitalize())

	split_results = []
	# 构造二元词组
	for i in range(3 * num_of_cheat_words):
		for j in range(3 * num_of_cheat_words):
			compound_words = modified_words[i] + modified_words[j]
			split = ""
			for k in range(len(modified_words[i])-1):
				split = split + "0"
			split = split + "1"
			for k in range(len(modified_words[j])-1):
				split = split + "0"
		split_results.append([compound_words, split])
	print(split_results)

# https://github.com/first20hours/google-10000-english.git
def preprocess_normal_english_words():
	cheat_file_csv= open(CHEAT_FILE, 'w', newline='')
	csvwriter = csv.writer(cheat_file_csv)

	all_words = open(GOOGLE_NORMAL_WORDS_FILE).readlines()
	words = []
	for word in all_words:
		word = word.strip().lower()
		# 跳过长度为小于等于2的单词
		if(len(word) > 2):
			words.append(word)

	all_identifiers = open(ORACLE_FILE).readlines()
	identifiers = []
	for identifier in all_identifiers:
		identifiers.append(identifier.split(' ')[1].lower())

	count = 1
	for i in range(len(words)):
		for j in range(len(identifiers)):
			if identifiers[j].find(words[i]) != -1:
				csvwriter.writerow([count, words[i]])
				count = count + 1
				break

if __name__ == '__main__':
	# preprocess_normal_english_words()
	trick_on_dataset()