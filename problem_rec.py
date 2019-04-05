"""
The code that is purposed to build up the recommendation problem
for the QandA development (specifically PundA)

We will try to implement the collaborative filtering technique for
giving up the recommendation to the user, which is expected to be the
problem the user is mostly interested in.
"""

import random
import csv
import numpy as np
import tensorflow as tf

#Step 1 : Preprocessing the data that is to be processed
########### DETAILS WILL BE UPDATED AFTER SPECIFIC INFO ABT DATA IS OBTAINED ##############

#Step 2 : Build up the network architecture
"""
Upon building up the network, we assume that the input data will consist of the user ID,
problem ID, whether the user has attempted that problem or not, whether the user answer the
problem correctly, and also the timestamp of the attempt (if the user has done one)

We plan to build up the machine that recommends user with the problem based on his/her inter-
action with the application during 2 weeks of his/her usage. We will inspect whether this
decision is correct or not, then we may change the duration later on
"""

#The first part is the problem input's feeding into network (Tensorflow)
matrix_input = tf.placeholder(tf.float32, [None, num_items])
attempted_input = tf.placeholder(tf.float32, [None, num_items])

nonzero_input = tf.math.count_nonzero(attempted_input, axis = 0)

#The second part is to define the parameters to be trained in this AutoEncoder
w_encoder_1 = tf.get_variable(name = 'w_encoder_1', shape = [num_items, hidden_1], initializer = tf.contrib.layers.xavier_initializer())
w_encoder_2 = tf.get_variable(name = 'w_encoder_2', shape = [hidden_1, hidden_2], initializer = tf.contrib.layers.xavier_initializer())

w_decoder_1 = tf.get_variable(name = 'w_decoder_1', shape = [hidden_2, hidden_1], initializer = tf.contrib.layers.xavier_initializer())
w_decoder_2 = tf.get_variable(name = 'w_decoder_2', shape = [hidden_1, num_items], initializer = tf.contrib.layers.xavier_initializer())

b_encoder_1 = tf.get_variable(name = 'b_encoder_1', shape = [hidden_1], initializer = tf.contrib.layers.xavier_initializer())
b_encoder_2 = tf.get_variable(name = 'b_encoder_2', shape = [hidden_2], initializer = tf.contrib.layers.xavier_initializer())

b_decoder_1 = tf.get_variable(name = 'b_decoder_1', shape = [hidden_1], initializer = tf.contrib.layers.xavier_initializer())
b_decoder_2 = tf.get_variable(name = 'b_decoder_2', shape = [num_items], initializer = tf.contrib.layers.xavier_initializer())

#The third part is to define all the operations to be done in this AE
layer_1 = tf.add(tf.matmul(matrix_input, w_encoder_1), b_encoder_1)
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, w_encoder_2), b_encoder_2)
layer_2 = tf.nn.relu(layer_2)

layer_3 = tf.add(tf.matmul(layer_2, w_decoder_1), b_decoder_1)
layer_3 = tf.nn.relu(layer_3)

matrix_output = tf.add(tf.matmul(layer_3, w_decoder_2), b_decoder_2)
#this matrix_output is the predicted rating by the user. the real target is the space in which
#user has attempted the problem (indicated by the value inside the attempted_input)

masked_input = tf.multiply(matrix_input, attempted_input)
masked_output = tf.multiply(matrix_output, attempted_input)
masked_diff = tf.div(tf.square(tf.subtract(masked_output, masked_input)), nonzero_input)

masked_mean = tf.reduce_mean(masked_diff)
loss = masked_mean + reg_param * (tf.nn.l2_loss(w_decoder_1) + tf.nn.l2_loss(w_decoder_2) \
	tf.nn.l2_loss(w_encoder_1) + tf.nn.l2_loss(w_encoder_2))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)