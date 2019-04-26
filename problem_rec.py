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
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_1', 128, 'The number of nodes in the first hidden layer')
flags.DEFINE_integer('hidden_2', 64, 'The number of nodes in the second hidden layer')
flags.DEFINE_boolean('train_mode', True, 'if True, system in training mode. Else, system in testing mode')
flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate of the network architecture')
flags.DEFINE_integer('training_epoch', 200, 'The number of training epoch')
flags.DEFINE_integer('batchSize', 64, 'Size of one batch for training and testing purpose')
flags.DEFINE_integer('sample_test', 0, '')

def map_grade(grade):
	if grade == "N":
		return 0.0
	elif grade == "S":
		return 1.0
	elif grade == "A":
		return 2.0
	elif grade == "B":
		return 3.0
	elif grade == "C":
		return 4.0
	else:
		return 5.0

input_file = [] #the list of grades user obtained in the respective genres
attempt_file = [] #the list of whether user has obtained grade in the respective genres

#Step 1 : Preprocessing the data that is to be processed
with open("user_grade_h1s1.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		input_file.append(row[1:])
		attempt_file.append(row[1:])

	input_file = input_file[1:]
	attempt_file = attempt_file[1:]

print("Sample input file (before conversion)\n", input_file[23])

#at this point, input_file and attempt_file both has the exact same dimensuibn
for i in range(len(attempt_file)):
	for j in range(len(attempt_file[0])):
		input_file[i][j] = map_grade(input_file[i][j])

		if attempt_file[i][j] == "N":
			attempt_file[i][j] = 0.0
		else:
			attempt_file[i][j] = 1.0

num_genre = len(input_file[0])
num_user = 0.01 * len(input_file)

print("Sample input file (after conversion)\n", input_file[23])

train_input, val_input, test_input = input_file[:int(0.7*num_user)], \
	input_file[int(0.7*num_user):int(0.85*num_user)], input_file[int(0.85*num_user):]
train_attempt, val_attempt, test_attempt = attempt_file[:int(0.7*num_user)], \
	attempt_file[int(0.7*num_user):int(0.85*num_user)], attempt_file[int(0.85*num_user):]

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
matrix_input = tf.placeholder(tf.float32, [None, num_genre])
attempted_input = tf.placeholder(tf.float32, [None, num_genre])

nonzero_input = tf.math.count_nonzero(attempted_input, axis = 0)
nonzero_input = tf.cast(nonzero_input, tf.float32)

#Define the hyperparameter for the network architecture
learning_rate = 1e-4
training_epoch = 100
batchSize = 64
reg_param = 0.3
hidden_1 = 128
hidden_2 = 64

# init = lambda shape, dtype: np.random.normal(loc = 0.0, scale = 2.0)

#The second part is to define the parameters to be trained in this AutoEncoder
with tf.device("/gpu:0"):
	w_encoder_1 = tf.get_variable(name = 'w_encoder_1', shape = [num_genre, FLAGS.hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	w_encoder_2 = tf.get_variable(name = 'w_encoder_2', shape = [FLAGS.hidden_1, FLAGS.hidden_2], initializer = tf.contrib.layers.xavier_initializer())
	w_decoder_1 = tf.get_variable(name = 'w_decoder_1', shape = [FLAGS.hidden_2, FLAGS.hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	w_decoder_2 = tf.get_variable(name = 'w_decoder_2', shape = [FLAGS.hidden_1, num_genre], initializer = tf.contrib.layers.xavier_initializer())

	b_encoder_1 = tf.get_variable(name = 'b_encoder_1', shape = [FLAGS.hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	b_encoder_2 = tf.get_variable(name = 'b_encoder_2', shape = [FLAGS.hidden_2], initializer = tf.contrib.layers.xavier_initializer())
	b_decoder_1 = tf.get_variable(name = 'b_decoder_1', shape = [FLAGS.hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	b_decoder_2 = tf.get_variable(name = 'b_decoder_2', shape = [num_genre], initializer = tf.contrib.layers.xavier_initializer())

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

	masked_sum = tf.reduce_sum(masked_diff)
	loss = tf.math.sqrt(masked_sum)

	optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(loss)

	init_op = tf.global_variables_initializer()

#The next process is to fetch all the data into its corresponding groups
#(either training, validation or testing data)

epoch_list, cost_list, cost_val_list = [], [], []

saver = tf.train.Saver()

if FLAGS.train_mode:
	with tf.Session() as sess:
		sess.run(init_op)

		for epoch in range(FLAGS.training_epoch):
			total_cost = 0.
			total_val_loss = 0.

			no_of_batches = int(len(train_input) / FLAGS.batchSize)
			no_of_batches_val = int(len(val_input) / FLAGS.batchSize)

			#Optimizing the network while also counting on the loss function of training set
			ptr = 0
			for i in range(no_of_batches):
				batch_input = train_input[ptr:ptr+FLAGS.batchSize]
				batch_attempt = train_attempt[ptr:ptr+FLAGS.batchSize]
				ptr += FLAGS.batchSize

				_, cost = sess.run([optimizer, loss], feed_dict = {matrix_input : batch_input, attempted_input : batch_attempt})
				total_cost += cost / no_of_batches

			#After the optimization, also count the loss function of validation set
			ptr = 0
			for i in range(no_of_batches_val):
				batch_input = val_input[ptr:ptr+FLAGS.batchSize]
				batch_attempt = val_attempt[ptr:ptr+FLAGS.batchSize]
				ptr += FLAGS.batchSize

				cost_val = sess.run(loss, feed_dict = {matrix_input : batch_input, attempted_input : batch_attempt})
				total_val_loss += cost_val / no_of_batches_val

			#Save the value for the loss graph creation
			epoch_list.append(epoch + 1)
			cost_list.append(total_cost)
			cost_val_list.append(total_val_loss)
			
			print("Finish Epoch# %d with Train Loss %.8f and Val Loss %.8f" % (epoch + 1, total_cost, total_val_loss))

		print("Optimization and Training Finished")

		saver.save(sess, "./model_h1s1_190426_1e4_data.ckpt")
		print("Pre-trained Model Saved")

		plt.plot(epoch_list, cost_list, "b", epoch_list, cost_val_list, "r")
		plt.xlabel("Epoch")
		plt.ylabel("Cost Function")

		plt.title("Rec System (Deep AE) Training")

		plt.savefig("AE_Training_190426_h1s1_1e4_data.png")

		plt.clf()

else:
	with tf.Session() as sess:
		saver.restore(sess, "model_h1s1_190426_1e4_data.ckpt")

		print("Saved Model has been Restored")

		#check on the testing dataset : its RMSE Loss and also its result
		no_of_batches_test = int(len(test_input) / FLAGS.batchSize)
		total_cost_test = 0.

		#First step : Calculating the RMSE of the testing dataset
		ptr = 0
		for i in range(no_of_batches_test):
			batch_input = test_input[ptr:ptr+FLAGS.batchSize]
			batch_attempt = test_input[ptr:ptr+FLAGS.batchSize]
			ptr += FLAGS.batchSize

			cost_test = sess.run(loss, feed_dict = {matrix_input : batch_input, attempted_input : batch_attempt})
			total_cost_test += cost_test / no_of_batches_test

		print("The Test Loss is %.8f" % (total_cost_test))

		#Second step : Showing one example of the test sample and its result
		test_sample = test_input[0]
		test_sample_a = test_attempt[0]
		result_sample = sess.run(matrix_output, feed_dict = {matrix_input : test_sample, attempted_input : test_sample_a})
		
		print("The sample test is", test_sample)
		print("The result test is", result_sample)