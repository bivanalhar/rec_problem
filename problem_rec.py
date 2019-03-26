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
database = []

with open("punda_data.csv") as punda:
	reader = csv.reader(punda, delimiter = ',')
	for row in reader:
		database.append(row) #collect all the user-problem information into var "database"

id_data = [i[0] for i in database]
prob_data = [i[1] for i in database]

num_user = len(set(id_data)) #number of users involved in the app usage
num_prob = len(set(prob_data)) #number of problems included in the database
########### DETAILS WILL BE UPDATED AFTER SPECIFIC INFO ABT DATA IS OBTAINED ##############

#Step 2 : Build up the network architecture

"""
In this network, we assume that the input will consist of number of users, number
of problems in the database, their interest in solving that problem, and whether they
have done the problem correctly or not

[we also keep on the record of whether the student has solved the problem correctly before]
(not sure if we will collect the type of type of the problems or just the problems itself)
"""

training_epoch = 100
learning_rate = 1e-3
batch_size = 256 #will adapt to the number of users and also the number of problems in dataset
hidden = 30 #number of hidden nodes in the network architecture
reg_param = 0.1

with tf.device("/gpu:0"):
	weight_user = tf.get_variable("weight_user", shape = [num_user, hidden], initializer = tf.contrib.layers.xavier_initializer())
	weight_prob = tf.get_variable("weight_prob", shape = [num_prob, hidden], initializer = tf.contrib.layers.xavier_initializer())
	bias_user = tf.get_variable("bias_user", shape = [num_user], initializer = tf.contrib.layers.xavier_initializer())
	bias_prob = tf.get_variable("bias_prob", shape = [num_prob], initializer = tf.contrib.layers.xavier_initializer())

	embd_user = tf.nn.embedding_lookup(weight_user, batch_user)
	embd_prob = tf.nn.embedding_lookup(weight_prob, batch_prob)

	embd_bias_user = tf.nn.embedding_lookup(bias_user, batch_user)
	embd_bias_prob = tf.nn.embedding_lookup(bias_prob, batch_prob)

	loss = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
	loss = tf.add(tf.add(loss, bias_user), bias_prob)
	loss = tf.nn.l2_loss(tf.subtract(loss, rate_batch))

	regularizer = reg_param * tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_prob))

	loss = tf.add(loss, regularizer)
	optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)