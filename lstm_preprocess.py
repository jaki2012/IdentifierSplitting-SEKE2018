from enchant.tokenize import get_tokenizer
from wordsegment import load, segment
import pandas as pd
import csv
import os
import itertools
import numpy as np
import time 
import string
import random
import re
from sklearn.preprocessing import LabelEncoder

CHEAT_FILE = "tmp/cheat_file.csv"
CHEAT_SPLITTING_FILE = "tmp/cheat_splitting_file1.csv"
CODED_FILE = "tmp/coded_file.csv"

RESULT_FILE = "tmp/final_result.csv"
EXPERI_DATA_PATH = "tmp/experi_data"
BT11_EXPERI_DATA_PATH = "tmp/nhs_bt11_experi_data"
EXPERI_RESULT_FILE = "tmp/experi_result.csv"
BT11_EXPERI_RESULT_FILE = "tmp/bt11_experi_result.csv"

df = pd.read_csv("tmp/non_hardsplit_bt11_oracle_samples.csv", header=None)
total_dict = df.values[:, 2:32]
total_dict_list = list(itertools.chain.from_iterable(total_dict))
sr_allwords = pd.Series(total_dict_list)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(0, len(set_words))
print(len(set_words))
tags = [ 'N', 'B', 'M', 'E', 'S']
tag_ids = range(len(tags))
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

def find_split_positions(seqs_list):
	positions = []
	for i in range(len(seqs_list) - 1):
		if seqs_list[i] == 'S' and seqs_list[i+1] == 'S':
			positions.append(i+1)
		elif seqs_list[i] == 'S' and seqs_list[i+1] == 'B':
			positions.append(i+1)
		elif seqs_list[i] == 'E' and seqs_list[i+1] == 'B':
			positions.append(i+1)
		elif seqs_list[i] == 'E' and seqs_list[i+1] == 'S':
			positions.append(i+1)
	if len(positions) != len(set(positions)):
		print("damn it !")
	return set(positions)

def cal_precison(verbose=False):
	df = pd.read_csv("tmp/final_result.csv", header=None)
	correct_answer = df.values[:, :30]
	total_result = df.values[:, 30:]
	lenresults = total_result.shape[0]
	labels = total_result[:, :30]
	logits = total_result[:, 30:]

	precision = 0
	recall = 0
	for j in range(lenresults):
		correct_splits = find_split_positions(labels[j])
		predict_splits = find_split_positions(logits[j])
		precise_splits = correct_splits & predict_splits
		precision = precision + (1 + len(precise_splits))/ (1+len(predict_splits))
		recall = recall + (1 + len(precise_splits))/ (1+len(correct_splits))


	precision = round((precision/lenresults),3)
	recall = round((recall/lenresults),3)
	fmesure = round( 2 * precision * recall / (precision + recall), 3)
	return fmesure

def cal_accuracy(verbose=False):
	df = pd.read_csv("tmp/final_result.csv", header=None)
	correct_answer = df.values[:, :30]
	total_result = df.values[:, 30:]
	lenresults = total_result.shape[0]
	labels = total_result[:, :30]
	logits = total_result[:, 30:]
	# 准确的数目
	right_sum = 0
	incorrect_index = []
	# print(lenresults)
	a = []
	for i in range(lenresults):
		a.append(''.join(correct_answer[i]))
	b= len(set(a))
	# print(len(a))
	print(b)
	# print(len(b))
	for j in range(lenresults):
		right = True
		list1 = labels[j]
		# print(list1)
		list2 = logits[j]
		# print(list2)
		# 准确率计算方式之一
		# 还有一种方式是利用字符串比较（截取有效字符串），后面的不管
		for i in range(len(list1)):
			right = True
			if(list1[i] == 'N'):
				if list2[i] != 'N':
					right = False
				break
			elif list1[i]!=list2[i]:
				right = False
				break;
		if right:
			right_sum  = right_sum +1
		else:
			incorrect_index.append(j)
			if verbose:
				print(''.join(correct_answer[j]))
				print(''.join(list1))
				print(''.join(list2))
	if verbose:
		print("Accuracy is %.3f" % (right_sum/lenresults))
		print("incorrect splitting %d / %d" %(len(incorrect_index), lenresults))
	return round((right_sum/lenresults),3)
	# for i in range(len(incorrect_index)):
	# 	print(correct_answer[incorrect_index[i]][0])
	# 	print(''.join(list1))
	# 	print(''.join(list2))

def vec2word(file):
	# df = pd.read_csv("tmp/sample1.csv", header=None)
	# total_dict = df.values[:, 2:]
	# lendict = total_dict.shape[0]
	# total_dict_list = list(itertools.chain.from_iterable(total_dict))
	# le = LabelEncoder()
	# le.fit(total_dict_list)

	# # data to be inverse_transform()
	# df1 = pd.read_csv("tmp/result1.csv", header=None)
	# # 不能要2
	# total_code = df1.values[:, :]
	# lencode = total_code.shape[0]
	# total_code_list = list(itertools.chain.from_iterable(total_code))
	# le_total_code_list = le.inverse_transform(total_code_list)
	# le_total_code = np.array(le_total_code_list).reshape(lencode,50)

	# final_result = np.column_stack((df.values[8699:, :2], le_total_code))
	result_csv = open(RESULT_FILE, 'w', newline='')
	csvwriter = csv.writer(result_csv)
	
	df1 = pd.read_csv(file, header=None)
	lendict = df1.values.shape[0]
	total_word_id_list = list(itertools.chain.from_iterable(df1.values[:, :30]))
	total_tag_id_list = list(itertools.chain.from_iterable(df1.values[:, 30:60]))
	total_tag_id_list1 = list(itertools.chain.from_iterable(df1.values[:, 60:]))
	words = coding(total_word_id_list, id2word)
	tags = coding(total_tag_id_list, id2tag)
	tags1 = coding(total_tag_id_list1, id2tag)
	csvwriter.writerows(np.column_stack((
		np.array(words).reshape(lendict,30), 
		np.array(tags).reshape(lendict,30), 
		np.array(tags1).reshape(lendict,30))))
	# csvwriter.writerows(final_result)

def word2vec(total_dict_list, dict_way=False):
	coded_file = open(CODED_FILE, 'w', newline='')
	csvwriter = csv.writer(coded_file)
	df = pd.read_csv("tmp/hardsplit_bt11_oracle_samples.csv", header=None)
	total_dict = df.values[:, 2:]
	lendict = total_dict.shape[0]
	if dict_way:
		total_dict_list = list(itertools.chain.from_iterable(total_dict))
		le = LabelEncoder()
		le.fit(total_dict_list)
		#获取vocabsize
		print(len(le.classes_))
		le_total_dict_list = le.transform(total_dict_list)
		le_total_dict = np.array(le_total_dict_list).reshape(lendict,60)
		csvwriter.writerows(le_total_dict)
	else:
		word_ids = coding(total_dict_list, word2id)
		total_tag_list = list(itertools.chain.from_iterable(df.values[:, 32:62]))
		tag_ids = coding(total_tag_list, tag2id)
		# print(tag_ids[:400])
		# print(lendict)
		# ddd = np.array(word_ids).reshape(lendict,30)
		# kkk = []
		# for i in range(20521):
		# 	kkk.append(ddd[i])
		# print(kkk)
		# print(len(set(kkk)))
		csvwriter.writerows(np.column_stack((np.array(word_ids).reshape(lendict,30), np.array(tag_ids).reshape(lendict,30))))

def coding(words, projection):
	ids = list(projection[words])
	return ids

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
			split = ['M'] * (len(modified_words[i]+modified_words[j]))
			# 产生一个0到3的位置
			random_position = random.randint(0,3)
			leni = len(modified_words[i])
			lenj = len(modified_words[j])
			if random_position == 0:
				compound_words = modified_words[i] + modified_words[j]
				splitted_words = modified_words[i] + '-' + modified_words[j]
				split[0] = 'B'
				split[leni-1] = 'E'
				split[leni] = 'B'
				split[leni+lenj-1] = 'E'
			elif random_position == 1:
				split.append('E')
				compound_words = splitter + modified_words[i] + modified_words[j]
				splitted_words = splitter + '-' + modified_words[i] + modified_words[j]
				split[0]='S'
				split[1]='B'
				split[leni]='E'
				split[leni+1]='B'
			elif random_position == 2:
				split.append('E')
				compound_words = modified_words[i] + splitter + modified_words[j]
				splitted_words = modified_words[i] + '-' + splitter + '-' + modified_words[j]
				split[0]='B'
				split[leni-1]='E'
				split[leni]='S'
				split[leni+1]='B'
			else:
				split.append('S')
				compound_words = modified_words[i] + modified_words[j] + splitter
				splitted_words = modified_words[i] + '-' + modified_words[j] + '-' + splitter
				split[0]='B'
				split[leni-1]='E'
				split[leni]='B'
				split[leni+lenj-1]='E'
			spare = 25 - len(compound_words)
			if spare > 0 :
				compound_word = compound_words
				for k in range(spare):
					compound_word = compound_word + ' '
					split.append('N')
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

def sort_experi_accuracies():
	results = []
	for train_option in ["pure_corpus", "mixed", "pure_oracle"]:
		for shuffle_option in ["True", "False"]:
			for cnn_option in range(1,4):
				accuracy = analyze_accuracy(train_option=train_option, shuffle_option=shuffle_option, cnn_option=cnn_option)
				results.append((train_option, shuffle_option, cnn_option, accuracy))

	results.sort(key=lambda x:x[3], reverse=True)
	for result in results:
		print(result)


def calculate_wordsegment_accuracy(verbose=False):

	def isNotSpecicalCharacter(part):
		if part in ['.', ':', '_', '~']:
			return False
		else:
			return True
	# wordsegment字典载入
	load()
	df = pd.read_csv("tmp/cheat_splitting_file.csv", header=None)
	identifiers = list(itertools.chain.from_iterable(df.values[34355:, 0:1]))
	splitted_identifiers = list(itertools.chain.from_iterable(df.values[34355:, 1:2]))
	lendata = len(identifiers)
	count = 0
	for i in range(lendata):
		wrong_split = True
		splitted_identifier = (splitted_identifiers[i]).lower()
		parts = splitted_identifier.split('-')
		condition = lambda part : part not in ['.', ':', '_', '~']
		parts = [x for x in filter(condition, parts)]
		wordsegmet_results = segment(identifiers[i])
		if len(parts) == len(wordsegmet_results):
			difference = list(set(parts).difference(set(wordsegmet_results)))
			if len(difference) == 0:
				count = count + 1
				wrong_split = False
		if verbose and wrong_split:
			print(parts)
			print(wordsegmet_results)

	print(count/lendata)



# train_option, cnn_option, shuffle
def analyze_accuracy(train_option=None, cnn_option=None, shuffle_option=None):
	if shuffle_option:
		shuffle_option = "True"
	else:
		shuffle_option = "False"
	experi_accuracies = []
	df = pd.read_csv("tmp/bt11_experi_result.csv", header=None)
	data = df.values
	lendata = len(data)
	count = 0
	for i in range(lendata):
		if train_option != None and train_option != data[i][1]:
			continue
		if cnn_option != None and cnn_option != data[i][2]:
			continue
		if shuffle_option != None and shuffle_option != str(data[i][3]):
			continue
		experi_accuracies.append(data[i][5])
		count = count + 1
	total_acc_score = 0
	for i in experi_accuracies:
		total_acc_score = total_acc_score + i

	return (total_acc_score/count)




def scan_experi_data():
	experi_results = []
	experi_results_csv = open(BT11_EXPERI_RESULT_FILE, 'w+', newline='')
	csvwriter = csv.writer(experi_results_csv)
	i = 0
	for path, subpaths, files in os.walk(BT11_EXPERI_DATA_PATH):
		for file in files:
			if os.path.join(path, file).find(".csv")==-1:
				continue
			vec2word(os.path.join(path, file))
			# print(os.path.join(path, file))
			accuracy = cal_accuracy()
			print(accuracy)
			# 后向断言
			pattern_train_option = re.compile(r'.*?(?=_cnn)')
			# 前向断言用search
			pattern_cnn_option = re.compile(r'(?<=_cnn).*(?=iter)')
			pattern_shuffle_option = re.compile(r'(?<=iter).*(?=bi)')
			pattern_iter_option = re.compile(r'(?<=iter)(\d)*(?=\S)')

			m1 = pattern_train_option.search(file)
			m2 = pattern_cnn_option.search(file)
			m3 = pattern_shuffle_option.search(file)
			m4 = pattern_iter_option.search(file)

			if m1:
				print("suck")
				i = i + 1
				# print(i)
				# look-behind requires fixed-width pattern in python 
				# print(m1.group(), m2.group(), m3.group().strip(string.digits), m4.group())
				# print(accuracy)
				# 转化为int才能排序
				print(i, m1.group(), m2.group(), m3.group().strip(string.digits), int(m4.group()), accuracy)
				experi_results.append((i, m1.group(), m2.group(), m3.group().strip(string.digits), int(m4.group()), accuracy))
				# csvwriter.writerow((i, m1.group(), m2.group(), m3.group().strip(string.digits), m4.group(), accuracy))

	for train_option in ["pure_corpus", "mixed", "pure_oracle"]:
		for shuffle_option in ["True", "False"]:
			for cnn_option in range(1, 4):
				temp_group = []
				for experi_result in experi_results:
					if train_option == experi_result[1] and shuffle_option == experi_result[3] and str(cnn_option) == experi_result[2]:
						temp_group.append(experi_result)
				temp_group.sort(key=lambda x: x[4])
				csvwriter.writerows(temp_group)


if __name__ == '__main__':
	# word2vec(total_dict_list)
	# vec2word()
	# cal_accuracy()
	# trick_on_dataset()
	# scan_experi_data()
	# sort_experi_accuracies()
	# calculate_wordsegment_accuracy(False)
	# print(analyze_accuracy(train_option="pure_corpus", cnn_option=2, shuffle_option=True))
	# find_split_positions(['S','B','M','M','E','B','M','M','E','S'])
	print(cal_precison())
	# print(cal_accuracy())

