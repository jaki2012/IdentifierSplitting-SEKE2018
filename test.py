import tensorflow as tf
import numpy as np

M = np.array([
        [[1],[-1],[0]],
        [[-1],[2],[1]],
        [[0],[2],[-2]]
    ])

print("Matrix shape is: ")
print( M.shape)

filter_weight = tf.get_variable('weights', [2, 3, 1, 1], initializer = tf.constant_initializer([
                                                                        [1, -1, 0],
                                                                        [2,1, 1]]))
biases = tf.get_variable('biases', [1], initializer = tf.constant_initializer(1))

M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)

x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides = [1, 1, 2, 1], padding = 'VALID')
bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias,feed_dict={x:M})
    pooled_M = sess.run(pool,feed_dict={x:M})
    filter_weight = sess.run(filter_weight)
    print(filter_weight)
    print(convoluted_M)
    # print("pooled_M: \n", pooled_M)