import tensorflow as tf
import numpy as np
import sys
import pandas as pd
import itertools
import getopt
from is_modeltrainning import * 
import os



chars = [' ', '$', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_ids = range(0, len(chars))
tags = [ 'N', 'B', 'M', 'E', 'S']
tag_ids = range(len(tags))
char2id = pd.Series(char_ids, index=chars)
id2char = pd.Series(chars, index=char_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

def splitting_by_tags(identi, tags):
	split_pos = [-1]
	terms = []
	# The spare N will be ignore
	for i in range(len(identi)-1):
		if(tags[i]=='E' and tags[i+1]=='B'):
			split_pos.append(i)
		elif tags[i]=='E' and tags[i+1]=='S':
			split_pos.append(i)
		elif tags[i]=='S' and tags[i+1]=='B':
			split_pos.append(i)
		elif tags[i]=='S' and tags[i+1]=='S':
			split_pos.append(i)
	split_pos.append(len(identi)-1)
	for i in range(len(split_pos)-1):
		term = identi[split_pos[i]+1:split_pos[i+1]+1]
		if(term is not '-'):
			terms.append(term)

	return terms


def coding(words, projection):
	ids = list(projection[words])
	return ids

def main(argv=None):

	# print(argv)
	# A Simple test case
	# a = splitting_by_tags("good-glLUCK1", ['B','M','M','E','S','B','E','B','M','M','E','S'])
	# print(a)

	try:
		opts, args = getopt.getopt(argv[1:], 'i:f:', ['help'])
	except getopt.GetoptError as err:
		print(str(err))

	test_identifiers = []
	for o, a in opts:
		if o in ["-i"]:
			test_identifiers = a.split(',')
		elif o in ["-f"]:
			df = pd.read_csv(a, header=None)
			test_identifiers= list(itertools.chain.from_iterable(df.values[:, 0:1]))


	
	# No identifiers to split
	if(len(test_identifiers) == 0):
		print("Receive no identifiers.. so exist..")
		return


	if not FLAGS.save_path:
		raise ValueError("Must set --save_path to language model directory")
	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 30

	FLAGS.cnn_option=2


	skip_data= []
	test_data= []
	for test_identifier in test_identifiers:
		# Skip those identifiers with length exceed 30
		if len(test_identifier) >30:
			skip_data.append(test_identifier)
			continue
		tocoded = [' ']* 30
		for i in range(len(test_identifier)):
			tocoded[i] = test_identifier[i]
		char_ids = coding(tocoded, char2id)
		test_data1 = char_ids
		for i in range(30):
			test_data1.append(1)
		test_data.append(test_data1)

	if len(skip_data) > 0:
		print("These identifiers are skipped because of the limitiation of length:")
		for i in range(len(skip_data)):
			print(skip_data[i])

	if len(test_data) == 0:
		print("Receive no identifiers.. so exist..")
		return
	
	test_data = np.array(test_data)
	
	initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
	test_queue = reader.ptb_producer(test_data, eval_config.batch_size, eval_config.num_steps)
	test_batch_len = len(test_data) // eval_config.batch_size

	with tf.name_scope("Train"):
		with tf.variable_scope("Model", reuse=None, initializer=initializer):
			mtest = PTBModel(is_trainning=False, config=eval_config)
	saver = tf.train.Saver()  
	
	with tf.Session() as session:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session,coord=coord)

		ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(session, ckpt.model_checkpoint_path)
		a = get_result(session, mtest, test_queue, None, False, test_batch_len)
		a = np.array(a)
		coord.request_stop()
		coord.join(threads)

		# gv = [v for v in tf.global_variables()]
		# i = 0
		# for v in gv:
		# 	print(v)
	
	a = a.reshape([-1, 1 , 90])

	print("Splitting results are shown below:")
	for i in range(len(a)):
		identi = ''.join(coding(a[i][0][0:30],id2char))
		tags = coding(a[i][0][60:90],id2tag)
		terms = splitting_by_tags(identi, tags)
		print(identi + "  ==>  " + '-'.join(terms))


if __name__ == "__main__":
	tf.app.run()