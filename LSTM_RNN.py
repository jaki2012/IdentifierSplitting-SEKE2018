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

flags.DEFINE_string(
	"data_path", "tmp/coded_file.csv",
	"The directory to retrive the data")

flags.DEFINE_bool(
	"use_fp16", False,
	"Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def get_feeddata(session, i, data, batch_size, num_steps):
	sequence = data[:, :25]
	label = data[:, -25:]
	x = tf.strided_slice(sequence, [i * batch_size, 0], [(i+1) * batch_size, num_steps])
	x.set_shape([batch_size, num_steps])
	y = tf.strided_slice(label, [i * batch_size, 0], [(i+1) * batch_size, num_steps])
	y.set_shape([batch_size, num_steps])
	return session.run([x, y])

def get_rawdata(path):
	df = pd.read_csv(path, header=None)
	data = df.values
	# 配置一
	train_data = data[:8499, 2:]
	valid_data = data[8499: 8699, 2:]
	test_data = data[8699:, 2:]

	# 配置二
	# train_data = data[:6699, 2:]
	# valid_data = data[6699: 6899, 2:]
	# test_data = data[6899:, 2:]
	return train_data, valid_data, test_data


def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float16

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

		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
		if is_trainning and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrpper(
				lstm_cell, output_keep_prob=config.keep_prob)
		# 多层lstm单元叠加起来
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
		self._initial_state = cell.zero_state(batch_size, data_type())

		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
			inputs = tf.nn.embedding_lookup(embedding, self._input_data)

		if is_trainning and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		output = tf.reshape(tf.concat(outputs,1 ), [-1, size])
		weight = tf.get_variable("weight", [size, vocab_size], dtype=data_type())

		bias = tf.get_variable("bias", [vocab_size], dtype=data_type())
		logits = tf.matmul(output, weight) + bias

		self._logits = logits

		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[tf.reshape(self._targets,[-1])],
			[tf.ones([batch_size * num_steps], dtype=data_type())])

		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state
		# 只在训练模型的时候定义BP操作
		if not is_trainning: return

		self._learning_rate = tf.Variable(0.0, trainable=False)
		trainable_variables = tf.trainable_variables()

		grads, _ = tf.clip_by_global_norm(
			tf.gradients(cost, trainable_variables), config.max_grad_norm)

		# 梯度下降优化，指定学习速率
		optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
		self._train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

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
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

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

def run_epoch(session, model, data, eval_op, verbose, epoch_size):
	# epoch_size = ((len(data) // model.batch_size) -1) // model.num_steps
	# epoch_size = (len(data) // model.batch_size) -1
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)
	# for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
	for i in range(epoch_size):
		# x,y = get_feeddata(session, i, data, model.batch_size, model.num_steps)
		x, y = session.run(data)
		fetches = [model.cost, model.final_state, eval_op]
		feed_dict = {}
		feed_dict[model.input_data] = x
		feed_dict[model.targets] = y
		cost, state, _ = session.run(fetches, feed_dict)

		costs += cost
		iters += model.num_steps
		if verbose and i % 100 == 0:
			print("After %d steps, perplexity is %.3f" % (i, np.exp(costs/iters)))
	return np.exp(costs/iters)

def get_result(session, model, data, eval_op, verbose, epoch_size):
	result_csv = open("tmp/result1.csv", 'w+')
	csvwriter= csv.writer(result_csv)
	batch_size = 20
	num_steps = 25
	for i in range(epoch_size):
		# 保证有序性
		# x, y = get_feeddata(session, i, data, batch_size, num_steps)
		x, y = session.run(data)
		fetches = [model.logits, model.input_data]
		feed_dict = {}
		feed_dict[model.input_data] = x
		feed_dict[model.targets] = y
		logits, input_data = session.run(fetches, feed_dict)
		predict = np.argmax(logits,2)
		predicts = []
		result =[]
		for j in range(25):
			predicts.append(predict[0][j])
		result.append(predicts)
		# 矩阵合并
		# 将原单词取回 避免多线程的打乱
		csvwriter.writerows(np.column_stack((input_data, y, result)))

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
		print("queue building finish")

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)
		for i in range(config.max_max_epoch):
			lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
			m.assign_lr(session, config.learning_rate * lr_decay)

			print("In iteration: %d" % (i+1))
			run_epoch(session, m, train_queue, m.train_op, True, train_batch_len)

			valid_perplexity = run_epoch(session, mValid, eval_queue, tf.no_op(), False, valid_batch_len)
			print("Epoch: %d Validation Perplexity: %.3f" % (i+1, valid_perplexity))
		test_perplexity = run_epoch(session, mTest, test_queue, tf.no_op(), False, test_batch_len)
		print("Final Test Perplexity: %.3f" % test_perplexity)
		get_result(session, mTest, test_queue, tf.no_op(), False, test_batch_len)
		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	tf.app.run()
