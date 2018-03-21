import tensorflow as tf
import os
import pandas as pd
import configparser
import itertools


def coding(words, projection):
	ids = list(projection[words])
	return ids


cf = configparser.ConfigParser()
cf.read('config.ini')
processing_project = "bt11_nhs_data"
CODED_FILE = cf.get(processing_project, "coded_file")
# print(CODED_FILE)
SAMEPLES_FILE = cf.get(processing_project, "oracle_samples_file")
with tf.Session() as session:
	signature_key = 'test_signature'

	input_key = 'input_x'
	output_key = 'output'

	meta_graph_def = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], os.path.join(os.getcwd(), 'model_save1/1'))

	# 从meta_graph_def中取出SignatureDef对象
	signature = meta_graph_def.signature_def
	# 从signature中找出具体输入输出的tensor name 
	x_tensor_name = signature[signature_key].inputs[input_key].name
	y_tensor_name = signature[signature_key].outputs[output_key].name

	# 获取tensor 并inference
	x = session.graph.get_tensor_by_name(x_tensor_name)
	y = session.graph.get_tensor_by_name(y_tensor_name)

	# _x 实际输入待inference的data
	a = session.run(y, feed_dict={x:[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]})

	df = pd.read_csv(SAMEPLES_FILE, header=None)
	total_dict = df.values[:, 2:32]
	total_dict_list = list(itertools.chain.from_iterable(total_dict))
	sr_allwords = pd.Series(total_dict_list)
	sr_allwords = sr_allwords.value_counts()
	set_words = sr_allwords.index
	set_ids = range(0, len(set_words))
	# print(len(set_words))
	tags = [ 'N', 'B', 'M', 'E', 'S']
	tag_ids = range(len(tags))
	word2id = pd.Series(set_ids, index=set_words)
	id2word = pd.Series(set_words, index=set_ids)
	tag2id = pd.Series(tag_ids, index=tags)
	id2tag = pd.Series(tags, index=tag_ids)

	words = coding(a[0], id2tag)

	print(words)