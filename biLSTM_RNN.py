# -*- coding:utf-8 -*-  
import tensorflow as tf
import numpy as np 
import reader
import pandas as pd
import time
import csv
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
	"model", "small",
	"A type of model. Possible options are: small, medium, large.")

flags.DEFINE_integer(
	"use_crf", 1,
	"whether to use crf layer. default yes")

flags.DEFINE_string(
	"data_path", "tmp/coded_file.csv",
	"The directory to retrive the data")

flags.DEFINE_bool(
	"use_fp16", False,
	"Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_string(
	"train_option", "pure_corpus",
	"The options detemining how to compose the data")

FLAGS = flags.FLAGS

def get_feeddata(session, i, data, batch_size, num_steps):
	sequence = data[:, :25]
	label = data[:, -25:]
	x = tf.strided_slice(sequence, [i * batch_size, 0], [(i+1) * batch_size, num_steps])
	x.set_shape([batch_size, num_steps])
	y = tf.strided_slice(label, [i * batch_size, 0], [(i+1) * batch_size, num_steps])
	y.set_shape([batch_size, num_steps])
	return session.run([x, y])

def get_sequence_lengths(x_inputs):
	sequence_lengths = [0] * len(x_inputs)
	count = 0
	for x_input in x_inputs:
		sequence_lengths[count] = np.count_nonzero(x_input)
		count = count + 1
	return sequence_lengths

def get_rawdata(path):
	df = pd.read_csv(path, header=None)
	data = df.values

	if FLAGS.train_option == "pure_corpus":
		# 配置一
		train_data = data[:6698, :]
		valid_data = data[6698: 7098, :]
		test_data = data[7098:, :]
	elif FLAGS.train_option == "mixed":
		# 配置二
		train_data = data[:10732, :]
		valid_data = data[10732: 10932, :]
		test_data = data[10932:, :]
	elif FLAGS.train_option == "pure_oracle":
		# 配置三
		train_data = data[6698:8498, :]
		valid_data = data[8498:8698, :]
		test_data = data[8698:, :]
	return train_data, valid_data, test_data

def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBModel(object):
	""" The PTB model """
	def __init__(self, is_trainning, config):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self._logits = []
		size = config.hidden_size
		vocab_size = config.vocab_size

		self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		self._targets = tf.placeholder(tf.int32,[batch_size, num_steps])
		self._sequence_lengths = tf.placeholder(tf.int32, shape=[batch_size])

		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
			inputs = tf.nn.embedding_lookup(embedding, self._input_data)

		if is_trainning and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
		lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
		if is_trainning and config.keep_prob < 1:
			lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, 
				output_keep_prob=config.keep_prob)
			lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, 
				output_keep_prob=config.keep_prob)
		# 多层lstm单元叠加起来
		cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layers, state_is_tuple=True)
		cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * config.num_layers, state_is_tuple=True)

		self._initial_state_fw = initial_state_fw = cell_fw.zero_state(batch_size, data_type())
		self._initial_state_bw = initial_state_bw = cell_bw.zero_state(batch_size, data_type())

		inputs = tf.unstack(inputs, num_steps, 1)

		outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
			initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)

		# outputs = []
		state_fw = self._initial_state_fw
		state_bw = self._initial_state_bw
		# with tf.variable_scope("RNN"):
		# 	for time_step in range(num_steps):
		# 		if time_step > 0: tf.get_variable_scope().reuse_variables()
		# 		(cell_output, state) = cell(inputs[:, time_step, :], state)
		# 		outputs.append(cell_output)
		# output = tf.reshape(tf.concat(outputs,1 ), [-1, size])
		output = tf.reshape(tf.concat(outputs, 1), [-1, size * 2])
		
		weight = tf.get_variable("weight", [size * 2, 5], dtype=data_type())
		bias = tf.get_variable("bias", [5], dtype=data_type())
		logits = tf.matmul(output, weight) + bias

		# if FLAGS.use_crf != 1:
		# 	pass
		# else:
		# 得分矩阵
		tag_scores = tf.reshape(logits, [batch_size, num_steps, 5])
		log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(tag_scores, self._targets, self._sequence_lengths)
		loss = tf.reduce_mean(-log_likelihood)
		# optimizer = tf.train.AdamOptimizer(lr)
		# train_op = optimizer.minimize(loss)

		viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(tag_scores, transition_params, self._sequence_lengths)
		self._viterbi_sequence = viterbi_sequence


		# self._logits = logits

		# loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
		# 	[logits],
		# 	[tf.reshape(self._targets,[-1])],
		# 	[tf.ones([batch_size * num_steps], dtype=data_type())])

		# self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._cost = cost = loss
		self._final_state_fw = state_fw
		self._final_state_bw = state_bw
		# 只在训练模型的时候定义BP操作
		if not is_trainning: return

		self._learning_rate = tf.Variable(0.0, trainable=False)
		trainable_variables = tf.trainable_variables()

		grads, _ = tf.clip_by_global_norm(
			tf.gradients(cost, trainable_variables), config.max_grad_norm)

		# 梯度下降优化，指定学习速率
		optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
		self._train_op = optimizer.apply_gradients(zip(grads,trainable_variables))
		# self._train_op = optimizer.minimize(loss)

		self._new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._learning_rate_update = tf.assign(self._learning_rate, self._new_learning_rate)


	def assign_lr(self, session, lr_value):
		# 使用 session 来调用 lr_update 操作
		session.run(self._learning_rate_update, feed_dict={self._new_learning_rate: lr_value})

	@property
	def input_data(self):
		return self._input_data

	@property
	def targets(self):
		return self._targets

	@property
	def logits(self):
		return self._logits

	@property
	def initial_state_fw(self):
		return self._initial_state_fw

	@property
	def initial_state_bw(self):
		return self._initial_state_bw

	@property
	def cost(self):
		return self._cost

	@property
	def viterbi_sequence(self):
		return self._viterbi_sequence

	@property
	def sequence_lengths(self):
		return self._sequence_lengths

	@property
	def final_state_fw(self):
		return self._final_state_fw

	@property
	def final_state_bw(self):
		return self._final_state_bw

	@property
	def lr(self):
		return self._learning_rate

	@property
	def train_op(self):
		return self._train_op

class SmallConfig(object):
	""" Small Config. """
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	# num_steps设置为单词的长度
	num_steps = 25
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 10
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	# default 100
	vocab_size = 67

def run_epoch(session, model, data, eval_op, verbose, epoch_size, Name="NOFOCUS"):
	# epoch_size = ((len(data) // model.batch_size) -1) // model.num_steps
	# epoch_size = (len(data) // model.batch_size) -1
	start_time = time.time()
	costs = 0.0
	iters = 0
	state_fw = session.run(model.initial_state_fw)
	state_bw = session.run(model.initial_state_bw)
	# for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
	for i in range(epoch_size):
		# x,y = get_feeddata(session, i, data, model.batch_size, model.num_steps)
		x, y = session.run(data)
		if Name == "hey2":
			print(x)
		fetches = [model.cost, model._final_state_fw, model._final_state_bw, eval_op]
		feed_dict = {}
		feed_dict[model.input_data] = x
		feed_dict[model.targets] = y
		feed_dict[model.sequence_lengths] = get_sequence_lengths(x)
		cost, state_fw, state_bw, _ = session.run(fetches, feed_dict)
		costs += cost
		iters += model.num_steps
		if verbose and i % 100 == 0:
			print("After %d steps, perplexity is %.3f" % (i, np.exp(costs/iters)))
	return np.exp(costs/iters)

def get_result(session, model, data, eval_op, verbose, epoch_size):
	result_csv_name = "tmp/" + FLAGS.train_option + '_' + 'biLSTMResult.csv'
	result_csv = open(result_csv_name, 'w+')
	csvwriter= csv.writer(result_csv)
	batch_size = 20
	num_steps = 25
	for i in range(epoch_size):
		# 保证有序性
		# x, y = get_feeddata(session, i, data, batch_size, num_steps)
		x, y = session.run(data)
		fetches = [model._viterbi_sequence, model.input_data]
		feed_dict = {}
		feed_dict[model.input_data] = x
		feed_dict[model.targets] = y
		feed_dict[model.sequence_lengths] = get_sequence_lengths(x)
		viterbi_sequence, input_data = session.run(fetches, feed_dict)
		# print(input_data)
		# print(logits)
		# predict = np.argmax(logits,1)
		# print(predict)
		# predicts = []
		# result =[]
		# for j in range(25):
		# 	predicts.append(predict[j])
		# result.append(predicts)
		# # 矩阵合并
		# # 将原单词取回 避免多线程的打乱
		csvwriter.writerows(np.column_stack((input_data, y, viterbi_sequence)))

def get_config():
	if FLAGS.model == "small":
		return SmallConfig()
	else:
		raise ValueError("Invalid model: %s", FLAGS.model)

def main(argv=None):
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to PTB data directory")
	print(FLAGS.data_path)
	# 获取原始数据
	# raw_data = reader.ptb_raw_data(FLAGS.data_path)
	# train_data, valid_data, test_data, _ = raw_data
	train_data, valid_data, test_data = get_rawdata(FLAGS.data_path)




	print("reading file finish")
	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 25

	# 计算一个epoch需要训练的次数
	train_data_len = len(train_data)
	train_batch_len = train_data_len // config.batch_size
	# train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP
	# print("train batch len: %d" % train_batch_len)


	valid_data_len = len(valid_data)
	valid_batch_len = valid_data_len // eval_config.batch_size
	# valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP
	

	test_data_len = len(test_data)
	test_batch_len = test_data_len // eval_config.batch_size
	# test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

	with tf.Graph().as_default(), tf.Session() as session:
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			m = PTBModel(is_trainning=True, config=config)
		with tf.variable_scope("model", reuse=True, initializer=initializer):
			mValid = PTBModel(is_trainning=False, config=eval_config)
			mTest = PTBModel(is_trainning=False, config=eval_config)

		tf.initialize_all_variables().run()

		train_queue = reader.ptb_producer(train_data, config.batch_size, config.num_steps)
		eval_queue = reader.ptb_producer(valid_data, eval_config.batch_size, eval_config.num_steps)
		test_queue = reader.ptb_producer(test_data, eval_config.batch_size, eval_config.num_steps)

		test_queue2 = reader.ptb_producer(test_data, eval_config.batch_size, eval_config.num_steps)
		print("queue building finish")

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)
		for i in range(config.max_max_epoch):
			lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
			m.assign_lr(session, config.learning_rate * lr_decay)

			print("In iteration: %d" % (i+1))
			run_epoch(session, m, train_queue, m.train_op, True, train_batch_len)

			valid_perplexity = run_epoch(session, mValid, eval_queue, tf.no_op(), False, valid_batch_len,"hey")
			print("Epoch: %d Validation Perplexity: %.3f" % (i+1, valid_perplexity))
		test_perplexity = run_epoch(session, mTest, test_queue, tf.no_op(), False, test_batch_len)
		print("Final Test Perplexity: %.3f" % test_perplexity)
		
		get_result(session, mTest, test_queue2, tf.no_op(), False, test_batch_len)
		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	tf.app.run()
