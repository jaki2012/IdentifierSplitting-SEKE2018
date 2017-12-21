from enchant.tokenize import get_tokenizer
import pandas as pd
import csv
import itertools
import numpy as np
import time 
import random
from sklearn.preprocessing import LabelEncoder

CHEAT_FILE = "tmp/cheat_file.csv"
CHEAT_SPLITTING_FILE = "tmp/cheat_splitting_file1.csv"
CODED_FILE = "tmp/coded_file.csv"

RESULT_FILE = "tmp/final_result.csv"

df = pd.read_csv("tmp/cheat_splitting_file.csv", header=None)
total_dict = df.values[:, 2:27]
total_dict_list = list(itertools.chain.from_iterable(total_dict))
sr_allwords = pd.Series(total_dict_list)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(0, len(set_words))
tags = [ 'N', 'B', 'M', 'E', 'S']
tag_ids = range(len(tags))
word2id = pd.Series(set_ids, index=set_words)
print(word2id)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

def cal_accuracy():
	df = pd.read_csv("tmp/final_result.csv", header=None)
	correct_answer = df.values[:, :25]
	total_result = df.values[:, 25:]
	lenresults = total_result.shape[0]
	labels = total_result[:, :25]
	logits = total_result[:, 25:]
	# 准确的数目
	right_sum = 0
	incorrect_index = []
	print(lenresults)
	a = []
	for i in range(lenresults):
		a.append(''.join(correct_answer[i]))
	b= list(set(a))
	print(len(b))
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
			print(''.join(correct_answer[j]))
			print(''.join(list1))
			print(''.join(list2))
		# 简易进度条
		# print("进度: ======={0}%".format(round((j + 1) * 100 / (lenresults))), end="\r")
		# time.sleep(0.01)
	print("Accuracy is %.3f" % (right_sum/lenresults))
	print("incorrect splitting %d / %d" %(len(incorrect_index), lenresults))
	# for i in range(len(incorrect_index)):
	# 	print(correct_answer[incorrect_index[i]][0])
	# 	print(''.join(list1))
	# 	print(''.join(list2))

def vec2word():
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
	
	df1 = pd.read_csv("tmp/pure_oracle_biLSTMResult.csv", header=None)
	lendict = df1.values.shape[0]
	total_word_id_list = list(itertools.chain.from_iterable(df1.values[:, :25]))
	total_tag_id_list = list(itertools.chain.from_iterable(df1.values[:, 25:50]))
	total_tag_id_list1 = list(itertools.chain.from_iterable(df1.values[:, 50:]))
	print(id2word)
	words = coding(total_word_id_list, id2word)
	tags = coding(total_tag_id_list, id2tag)
	tags1 = coding(total_tag_id_list1, id2tag)
	csvwriter.writerows(np.column_stack((
		np.array(words).reshape(lendict,25), 
		np.array(tags).reshape(lendict,25), 
		np.array(tags1).reshape(lendict,25))))
	# csvwriter.writerows(final_result)

def word2vec(total_dict_list, dict_way=False):
	coded_file = open(CODED_FILE, 'w', newline='')
	csvwriter = csv.writer(coded_file)
	df = pd.read_csv("tmp/cheat_splitting_file.csv", header=None)
	total_dict = df.values[:, 2:]
	lendict = total_dict.shape[0]
	if dict_way:
		total_dict_list = list(itertools.chain.from_iterable(total_dict))
		le = LabelEncoder()
		le.fit(total_dict_list)
		#获取vocabsize
		print(len(le.classes_))
		le_total_dict_list = le.transform(total_dict_list)
		le_total_dict = np.array(le_total_dict_list).reshape(lendict,50)
		csvwriter.writerows(le_total_dict)
	else:
		word_ids = coding(total_dict_list, word2id)
		total_tag_list = list(itertools.chain.from_iterable(df.values[:, 27:52]))
		tag_ids = coding(total_tag_list, tag2id)
		# print(tag_ids[:400])
		csvwriter.writerows(np.column_stack((np.array(word_ids).reshape(lendict,25), np.array(tag_ids).reshape(lendict,25))))

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
if __name__ == '__main__':
	# word2vec(total_dict_list)
	vec2word()
	cal_accuracy()
	# trick_on_dataset()
	# print()