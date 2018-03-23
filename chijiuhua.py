import tensorflow as tf
import numpy as np
from biLSTM_RNN import * 
import os


# 只有编码一致，才能确保模型输入一致。才能确保输出结果正确
chars = [' ', '$', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_ids = range(0, len(chars))
print(len(char_ids))
tags = [ 'N', 'B', 'M', 'E', 'S']
tag_ids = range(len(tags))
char2id = pd.Series(char_ids, index=chars)
id2char = pd.Series(chars, index=char_ids)
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
		char_ids = coding(tocoded, char2id)
		test_data1 = char_ids
		for i in range(30):
			test_data1.append(1)
		test_data.append(test_data1)
	
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
		print(ckpt)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(session, ckpt.model_checkpoint_path)
			print("hahah")
		a = get_result(session, mtest, test_queue, None, False, test_batch_len)
		a = np.array(a)
		coord.request_stop()
		coord.join(threads)

		# gv = [v for v in tf.global_variables()]
		# i = 0
		# for v in gv:
		# 	# if(i==1):
		# 	print(v)
			# i=i+1
	a = a.reshape([-1, 1 , 90])
	print(''.join(coding(a[0][0][60:90],id2tag)))
	print(''.join(coding(a[1][0][60:90],id2tag)))
	print(''.join(coding(a[2][0][60:90],id2tag)))

if __name__ == "__main__":
	tf.app.run()