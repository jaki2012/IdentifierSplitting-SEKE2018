import tensorflow as tf
import numpy as np
from biLSTM_RNN import * 
import os


# 目前仍然是不固定
total_dict_list = [' ', '$', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
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

def coding(words, projection):
	ids = list(projection[words])
	return ids

def main(argv=None):

	if not FLAGS.save_path:
		raise ValueError("Must set --save_path to language model directory")
	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 30

	FLAGS.cnn_option=2

	test_identifiers = ["goodLuckPaper","acceptMyPaper","goodLuckPaper1"]
	test_data= []
	for test_identifier in test_identifiers:
		tocoded = [' ']* 30
		for i in range(len(test_identifier)):
			tocoded[i] = test_identifier[i]
		word_ids = coding(tocoded, word2id)
		test_data1 = word_ids
		for i in range(30):
			test_data1.append(1)
		test_data.append(test_data1)
	
	test_data = np.array(test_data)
	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
		test_queue = reader.ptb_producer(test_data, eval_config.batch_size, eval_config.num_steps)
		test_batch_len = len(test_data) // eval_config.batch_size

		with tf.name_scope("Test"):
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = PTBModel(is_trainning=False, config=eval_config)

		b = FLAGS.save_path
		print(b)
		sv = tf.train.Supervisor()
		print(FLAGS.save_path)
		with sv.managed_session() as session:
			
			
			ckpt = tf.train.get_checkpoint_state(b)
			print(ckpt)
			if ckpt and ckpt.model_checkpoint_path:
				sv.saver.restore(session, ckpt.model_checkpoint_path)
				a = get_result(session, mtest, test_queue, None, False, test_batch_len)
				a = np.array(a)
		sess = tf.Session()
		gv = [v for v in tf.global_variables()]
		for v in gv:
			print(v.eval(session=sess))
	# a = a.reshape([-1, 1 , 90])
	# print(coding(a[0][0][60:90],id2tag))
	# print(coding(a[1][0][60:90],id2tag))
	# print(coding(a[2][0][60:90],id2tag))

if __name__ == "__main__":
	tf.app.run()