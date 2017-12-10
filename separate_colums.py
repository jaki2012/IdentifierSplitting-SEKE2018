import numpy as np
import pandas as pd
import csv
import sys
import os
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt  

# 训练集
TRAIN_FILE = "/Users/lijiechu/Desktop/newRule.dat"
CSV_TRAIN_FILE = "/Users/lijiechu/Desktop/newRule.csv"
TRAIN_FEATURES_CSV = "/Users/lijiechu/Desktop/train.features"
TRAIN_LABELS_CSV = "/Users/lijiechu/Desktop/train.labels"

# 测试集
TEST_FILE = "/Users/lijiechu/Desktop/newCheck.dat"
CSV_TEST_FILE = "/Users/lijiechu/Desktop/newTest.csv"
TEST_FEATURES_CSV = "/Users/lijiechu/Desktop/test.features"
TEST_LABELS_CSV = "/Users/lijiechu/Desktop/test.labels"

# svmlin文件目录
SVMLIN_PATH = "/Users/lijiechu/Documents/svmlin-v1.0/"

# 执行tsvm
def execute_tsvm():
	train_cmd = SVMLIN_PATH + "svmlin -A 2 " + TRAIN_FEATURES_CSV + " " + TRAIN_LABELS_CSV
	train_info = os.popen(train_cmd).readlines()
	for line in train_info:
		stripped_line = line.strip('\n')
		print(stripped_line)

	evaluate_cmd = SVMLIN_PATH + "svmlin -f train.features.weights " + TEST_FEATURES_CSV + " " + TEST_LABELS_CSV
	print(evaluate_cmd)
	evaluate_info = os.popen(evaluate_cmd).readlines()
	for line in evaluate_info:
		stripped_line = line.strip('\n')
		print(stripped_line)

# 预处理dat文件
def preprocess_dat(datfile, csvfile):
	csv_file = open(csvfile, 'w', newline='')
	csv_writer = csv.writer(csv_file)
	file = open(datfile)

	for line in file.readlines():
		prepocessed_line = line.strip('\n').split('\t')
		csv_writer.writerow(prepocessed_line)

def preprocess_csv(data_csv, features_csv, labels_csv, remainingLast=True):
	df = pd.read_csv(data_csv, header=None)

	data = df.values
	print(type(data))
	# 根据第一列的值排序，用于辅助挑选训练样本
	# data = data[np.lexsort(data[:,::-1].T)]
	trainning_labels = data[:, 0]
	trainning_features = data[:, 1:]

	with open(labels_csv, 'w+') as labels:
		temp_labels = ''
		for i in range(trainning_labels.size):
			temp_labels = temp_labels + str(trainning_labels[i]) + '\n'
		if False == remainingLast:
			temp_labels = temp_labels[ : -1]
		labels.write(temp_labels)

	with open(features_csv, 'w+') as features:
		temp_features = ''
		for i in range(trainning_labels.size):
			feature = ''
			for j in range(trainning_features[i].size):
				feature =  feature + trainning_features[i][j] + ' ';
			feature = feature + '\n'
			temp_features = temp_features + feature
		if False == remainingLast:
			temp_features = temp_features[ : -1]
		features.write(temp_features)

def evaluation_results(true_csv, pred_csv, ploting=False, threshold=0.5):
	true_df = pd.read_csv(true_csv,header=None)
	true_values = list(itertools.chain.from_iterable(true_df.values))
	pred_df = pd.read_csv(pred_csv,header=None)
	pred_values = list(itertools.chain.from_iterable(pred_df.values))
	
	# 绘制ROC曲线并计算AUC值（根据Threshold）
	if ploting:
		print(roc_auc_score(true_values, pred_values))

		fpr, tpr, thresholds = roc_curve(true_values, pred_values, pos_label=1)
		roc_auc = auc(fpr, tpr)  
		plt.plot(fpr, tpr, lw=2, label='ROC curve(area = %0.2f)' % ( roc_auc))
		plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random')
		plt.xlabel('False Positive Rate')  
		plt.ylabel('True Positive Rate')  
		plt.title('Receiver operating characteristic example') 
		plt.show()
	# accuracy = 0
	# for i in range(len(true_values)):
	# 	if(true_values[i] * pred_values[i] >0):
	# 		accuracy = accuracy + 1
	# print("Accuracy = %g" % (accuracy / len(true_values)))
	pred_values = [1 if x > threshold else -1 for x in pred_values]
	# p = 0
	# tp = 0
	# n = 0
	# tn = 0
	# for i in range(len(true_values)):
	# 	if true_values[i] == -1:
	# 		n = n + 1
	# 		if pred_values[i] == -1:
	# 			tn = tn + 1
	# 	else:
	# 		p = p + 1
	# 		if pred_values[i] == 1:
	# 			tp = tp + 1
	# print("p is %d, tp is %d, n is %d, tn %d" % (p, tp, n, tn))

	# p = 0
	# tp = 0
	# n = 0
	# tn = 0
	# for i in range(len(true_values)):
	# 	if pred_values[i] == -1:
	# 		n = n + 1
	# 		if true_values[i] == -1:
	# 			tn = tn + 1
	# 	else:
	# 		p = p + 1
	# 		if true_values[i] == 1:
	# 			tp = tp + 1
	# print("p is %d, tp is %d, n is %d, tn %d" % (p, tp, n, tn))
	
	target_names = ['Non-Defective', 'Defective']
	print(classification_report(true_values, pred_values, target_names=target_names))

if __name__ == "__main__":
	if 1 == len(sys.argv):
		# 预处理训练集
		preprocess_dat(TRAIN_FILE, CSV_TRAIN_FILE)
		preprocess_csv(CSV_TRAIN_FILE, TRAIN_FEATURES_CSV, TRAIN_LABELS_CSV)
		# 预处理测试集
		preprocess_dat(TEST_FILE, CSV_TEST_FILE)
		preprocess_csv(CSV_TEST_FILE, TEST_FEATURES_CSV, TEST_LABELS_CSV)
		execute_tsvm()
	else:
		# 评估数据集
		evaluation_results(TEST_LABELS_CSV, "test.features.outputs", False, 0)