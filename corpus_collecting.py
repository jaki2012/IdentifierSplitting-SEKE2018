from enchant.tokenize import get_tokenizer
import pandas as pd
import csv
import time 
import random

tknzr = get_tokenizer("en_US")
texts = [w for w in tknzr("thisissomesimpletext")]

ORACLE_FILE = "/Users/lijiechu/Desktop/loyola-udelaware-identifier-splitting-oracle.txt"
GOOGLE_NORMAL_WORDS_FILE = "/Users/lijiechu/Documents/google-10000-english/google-10000-english-no-swears.txt"
CHEAT_FILE = "tmp/cheat_file.csv"
CHEAT_SPLITTING_FILE = "tmp/cheat_splitting_file.csv"

def trick_on_dataset():
	cheat_splitting_file_csv= open(CHEAT_SPLITTING_FILE, 'w', newline='')
	csvwriter = csv.writer(cheat_splitting_file_csv)
	df = pd.read_csv(CHEAT_FILE, header=None, keep_default_na=False)
	data = df.values
	# 这样裁剪出来的依然是二维的
	cheat_words = data[:, 1:]
	num_of_cheat_words = len(cheat_words)
	# 加入大小写，首字母大写等形态
	modified_words = []
	for i in range(num_of_cheat_words):
		cheat_word = cheat_words[i][0]
		modified_words.append(cheat_word.lower())
		modified_words.append(cheat_word.upper())
		modified_words.append(cheat_word.capitalize())

	split_results = []
	splitter = '_'
	# 构造二元词组
	# 将4000万的数据量变为约1000万
	# 1000万数据依然能达到5.85GB
	for i in range(0, 3 * num_of_cheat_words, 2):
		for j in range(1, 3 * num_of_cheat_words, 2):
			split = [0] * (len(modified_words[i]+modified_words[j]) - 1)
			# 产生一个0到3的位置
			random_position = random.randint(0,3)
			# TODO:需要用优雅的方式处理
			# 方法一：利用字符串下标
			# 利用统一式子
			if random_position == 0:
				compound_words = modified_words[i] + modified_words[j]
				splitted_words = modified_words[i] + '-' + modified_words[j]
				split[len(modified_words[i])] = 1
			elif random_position == 1:
				split.append(0)
				compound_words = splitter + modified_words[i] + modified_words[j]
				splitted_words = splitter + '-' + modified_words[i] + modified_words[j]
				split[0] = 1
				split[len(modified_words[i])+1] = 1
			elif random_position == 2:
				split.append(0)
				compound_words = modified_words[i] + splitter + modified_words[j]
				splitted_words = modified_words[i] + '-' + splitter + '-' + modified_words[j]
				split[len(modified_words[i])] = 1
				split[len(modified_words[i])+1] = 1
			else:
				split.append(0)
				compound_words = modified_words[i] + modified_words[j] + splitter
				splitted_words = modified_words[i] + '-' + modified_words[j] + '-' + splitter
			spare = 25 - len(compound_words)
			if spare > 0 :
				compound_word = compound_words
				for k in range(spare):
					compound_word = compound_word + ' '
					split.append(0)
				# split_results.append([compound_words, splitted_words, list(''.join(list(compound_word))) , list(''.join(str(split))) ])
				words = []
				words.append(compound_words)
				words.append(splitted_words)
				csvwriter.writerow(words + list(''.join(list(compound_word))) + split)		
		# 简易进度条
		print("进度: ======={0}%".format(round((i + 1) * 100 / (3*num_of_cheat_words))), end="\r")
		time.sleep(0.01)
	# csvwriter.writerows(split_results)
	# print(split_results)

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