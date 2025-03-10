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
import pickle

matplotlib.use('Agg')

import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_1', 128, 'The number of nodes in the first hidden layer')
flags.DEFINE_integer('hidden_2', 64, 'The number of nodes in the second hidden layer')
flags.DEFINE_boolean('train_mode', True, 'if True, system in training mode. Else, system in testing mode')
flags.DEFINE_integer('batchSize', 64, 'Size of one batch for training and testing purpose')
flags.DEFINE_integer('test_set', 1, "Defining the set that is to be tested for the performance evaluation")

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

def map_attempt(grade):
	if grade == "N":
		return 0.0
	else:
		return 1.0

def map_inverse_attempt(grade):
	if grade != "N":
		return 0.0
	else:
		return 1.0

def convert_grade(number):
	if number <= 1.5:
		return 1.0
	elif number <= 2.5:
		return 2.0
	elif number <= 3.5:
		return 3.0
	elif number <= 4.5:
		return 4.0
	else:
		return 5.0

vec_convert = np.vectorize(convert_grade)
vec_grade = np.vectorize(map_grade)
vec_attempt = np.vectorize(map_attempt)
vec_inverse = np.vectorize(map_inverse_attempt)

input_train, input_val, input_test = [], [], [] #the list of grades user obtained in the respective genres
attempt_train, attempt_val, attempt_test = [], [], [] #the list of whether user has obtained grade in the respective genres

#Step 1 : Preprocessing the train, val and test data
with open("train_data_modified.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		input_train.append(row[2:]) #the rest of the data is the details about user's grade
		attempt_train.append(row[2:])

with open("val_data_modified.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		input_val.append(row[2:])
		attempt_val.append(row[2:])

with open("test_data_modified.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		input_test.append(row[2:])
		attempt_test.append(row[2:])

#print("Sample input file (before conversion)\n", input_file[23])

#at this point, input_file and attempt_file both has the exact same dimension
input_train, input_val, input_test = vec_grade(input_train), \
	vec_grade(input_val), vec_grade(input_test)
attempt_train, attempt_val, attempt_test = vec_attempt(attempt_train), \
	vec_attempt(attempt_val), vec_attempt(attempt_test)

num_genre = len(input_train[0])

#print("Sample input file (after conversion)\n", input_file[23])
#print("Sample attempted file\n", attempt_file[23])

#Step 2 : Build up the network architecture
"""
Upon building up the network, we assume that the input data will consist of the user ID,
problem ID, whether the user has attempted that problem or not, whether the user answer the
problem correctly, and also the timestamp of the attempt (if the user has done one)

We plan to build up the machine that recommends user with the problem based on his/her inter-
action with the application during 2 weeks of his/her usage. We will inspect whether this
decision is correct or not, then we may change the duration later on
"""

print("The number of data in Training Dataset is", len(input_train))
print("The number of data in Validation Dataset is", len(input_val))
print("The number of data in Testing Dataset is", len(input_test))

#The first part is the problem input's feeding into network (Tensorflow)
matrix_input = tf.placeholder(tf.float32, [None, num_genre])
attempted_input = tf.placeholder(tf.float32, [None, num_genre])

nonzero_input = tf.math.count_nonzero(attempted_input)
nonzero_input = tf.cast(nonzero_input, tf.float32)

#Define the hyperparameter for the network architecture
learning_rate = tf.placeholder(tf.float32, [])
training_epoch = 1000
batchSize = 64
hidden_1 = 128
hidden_2 = 64

#The second part is to define the parameters to be trained in this AutoEncoder
with tf.device("/gpu:0"):
	w_encoder_1 = tf.get_variable(name = 'w_encoder_1', shape = [num_genre, hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	w_encoder_2 = tf.get_variable(name = 'w_encoder_2', shape = [hidden_1, hidden_2], initializer = tf.contrib.layers.xavier_initializer())
	w_decoder_1 = tf.get_variable(name = 'w_decoder_1', shape = [hidden_2, hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	w_decoder_2 = tf.get_variable(name = 'w_decoder_2', shape = [hidden_1, num_genre], initializer = tf.contrib.layers.xavier_initializer())

	b_encoder_1 = tf.get_variable(name = 'b_encoder_1', shape = [hidden_1], initializer = tf.contrib.layers.xavier_initializer())
	b_encoder_2 = tf.get_variable(name = 'b_encoder_2', shape = [hidden_2], initializer = tf.contrib.layers.xavier_initializer())
	b_decoder_1 = tf.get_variable(name = 'b_decoder_1', shape = [hidden_1], initializer = tf.contrib.layers.xavier_initializer())
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

	masked_diff = tf.reduce_sum(tf.square(tf.subtract(masked_output, masked_input)))
	loss = tf.math.sqrt(tf.div(masked_diff, nonzero_input))

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	init_op = tf.global_variables_initializer()

#The next process is to fetch all the data into its corresponding groups
#(either training, validation or testing data)

#epoch_list, cost_list, cost_val_list = [], [], []

epoch_save_list = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
learning_rate_list = [3e-3, 1e-3, 5e-4, 3e-4, 1e-4]
batchSize = FLAGS.batchSize
hidden_1 = FLAGS.hidden_1
hidden_2 = FLAGS.hidden_2

if FLAGS.train_mode:
	pickle_out = open("epoch_lr_mix.pickle", "wb")

	#to be filled in with the best epoch and learning rate during training session
	best_loss = None
	best_epoch = None
	best_lr = None

	for learn_rate in learning_rate_list:
		epoch_list, cost_list, cost_val_list = [], [], []
		saver = tf.train.Saver()
		
		with tf.Session() as sess:
			sess.run(init_op)

			for epoch in range(training_epoch):
				total_cost = 0.
				total_val_loss = 0.

				no_of_batches = int(len(input_train) / batchSize)
				no_of_batches_val = int(len(input_val) / batchSize)

				#Optimizing the network while also counting on the loss function of training set
				ptr = 0
				for i in range(no_of_batches):
					batch_input = input_train[ptr:ptr+batchSize]
					batch_attempt = attempt_train[ptr:ptr+batchSize]
					ptr += batchSize

					_, cost = sess.run([optimizer, loss], feed_dict = {matrix_input : batch_input, attempted_input : batch_attempt, learning_rate : learn_rate})
					total_cost += cost / no_of_batches

				#After the optimization, also count the loss function of validation set
				ptr = 0
				for i in range(no_of_batches_val):
					batch_input = input_val[ptr:ptr+batchSize]
					batch_attempt = attempt_val[ptr:ptr+batchSize]
					ptr += batchSize

					cost_val = sess.run(loss, feed_dict = {matrix_input : batch_input, attempted_input : batch_attempt, learning_rate : learn_rate})
					total_val_loss += cost_val / no_of_batches_val

				#Save the value for the loss graph creation
				epoch_list.append(epoch + 1)
				cost_list.append(total_cost)
				cost_val_list.append(total_val_loss)
				
				if epoch % 50 == 49:
					print("Finish Epoch# %d with Train Loss %.8f and Val Loss %.8f" % (epoch + 1, total_cost, total_val_loss))

				if epoch in epoch_save_list:
					if best_loss == None or best_loss > total_val_loss:
						saver.save(sess, "./model_h1s1_1e5_mix_data.ckpt")
						best_epoch, best_lr = epoch + 1, learn_rate
						best_loss = total_val_loss
						print("The current best loss value is %.8f, achieved with epoch %d and learning rate %.0e" % \
							(best_loss, best_epoch, best_lr))
					
			print("Optimization and Training Finished")

			plt.plot(epoch_list, cost_list, "b", epoch_list, cost_val_list, "r")
			plt.xlabel("Epoch")
			plt.ylabel("Cost Function")

			plt.title("Rec System (Deep AE) Training")

			plt.savefig("AE_Training_h1s1_1e5_mix_data_lr_%.0e.png" % (learn_rate))

			plt.clf()

	dump_set = [best_loss, best_epoch, best_lr]
	print("The best loss for the network is %.8f" % (best_loss))
	print("The best epoch and learning rate is %d, %.0e" % (best_epoch, best_lr))
	pickle.dump(dump_set, pickle_out)
	pickle_out.close()

else:
	pickle_in = open("epoch_lr_mix.pickle", "rb")
	dumped_set = pickle.load(pickle_in)
	print("The set of best hyperparameter is %d, %.0e" % (dumped_set[1], dumped_set[2]))

	#The first phase of the testing session is to test on the splitted test data (from the distribution)
	with tf.Session() as sess:
		saver = tf.train.Saver()

		file_name = "model_h1s1_1e5_mix_data.ckpt"
		saver.restore(sess, file_name)

		print("Saved Model has been Restored")

		#check on the testing dataset : its RMSE Loss, its result and overall result analysis
		no_of_batches_test = int(len(input_test) / batchSize)
		total_cost_test = 0.

		#First step : Calculating the RMSE of the testing dataset
		ptr = 0
		for i in range(no_of_batches_test):
			batch_input = input_test[ptr:ptr+batchSize]
			batch_attempt = attempt_test[ptr:ptr+batchSize]
			ptr += batchSize

			cost_test = sess.run(loss, feed_dict = {matrix_input : batch_input, \
				attempted_input : batch_attempt, learning_rate : dumped_set[2]})
			total_cost_test += cost_test / no_of_batches_test

		print("The Validation Loss is %.8f" % (dumped_set[0]))
		print("The Test Loss is %.8f\n" % (total_cost_test))

		#Second step : Showing one example of the test sample and its result
		test_sample = np.reshape(input_test[1739+2675+263], [1, num_genre])
		test_sample_a = np.reshape(attempt_test[1739+2675+263], [1, num_genre])
		result_sample = sess.run(matrix_output, feed_dict = {matrix_input : test_sample, \
			attempted_input : test_sample_a, learning_rate : dumped_set[2]})
		
		print("Showing one example of the test sample")
		print("The test sample is\n", test_sample)
		print("and the result is\n", vec_convert(result_sample))

		#Third step : Counting on the average of the difference after conversion
		ptr = 0
		overall_diff = 0.
		for i in range(no_of_batches_test):
			batch_input = input_test[ptr:ptr+batchSize]
			batch_attempt = attempt_test[ptr:ptr+batchSize]
			ptr += batchSize

			matrix_test = sess.run(matrix_output, feed_dict = {matrix_input : batch_input, \
				attempted_input : batch_attempt, learning_rate : dumped_set[2]})
			matrix_test = vec_convert(matrix_test)
			#for j in range(len(matrix_test)):
			#	for k in range(len(matrix_test[j])):
			#		matrix_test[j][k] = convert_grade(matrix_test[j][k])
			nonzero_count = np.float32(np.count_nonzero(batch_attempt))

			matrix_square_diff = np.multiply(np.square(np.subtract(matrix_test, batch_input)), batch_attempt)
			total_square_diff = np.sum(matrix_square_diff)
			avg_diff_test = np.sqrt(np.divide(total_square_diff, nonzero_count))
			overall_diff += avg_diff_test / no_of_batches_test
		
		print("The Overall Difference in Test Data is %.8f" % (overall_diff))

	#The second phase of the testing session is to test the network from the independent dataset
	with tf.Session() as sess:
		saver = tf.train.Saver()

		file_name = "model_h1s1_1e5_mix_data.ckpt"
		saver.restore(sess, file_name)

		print("Saved Model has been Revived")

		#first, we need to load the testing file
		real_input, test_input, test_attempt, test_inverse = [], [], [], []
		with open("test_grade_h1s1_%d.csv" % (FLAGS.test_set)) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter = ",")
			for row in csv_reader:
				real_input.append(row[2:])

		for i in range(len(real_input)):
			#second, we determine the percentage of non-obtained genre
			random_N_genre = int(num_genre * (random.choice(range(21)) + 70) / 100)
			list_N_genre = list(range(num_genre))
			random.shuffle(list_N_genre)

			list_N_genre = list_N_genre[:random_N_genre]

			#third, we designate the grade to each non-obtained genre with N and real grade otherwise
			list_grade_genre = []
			for j in range(len(real_input[i])):
				if j in list_N_genre:
					list_grade_genre.append("N")
				else:
					list_grade_genre.append(real_input[i][j])

			test_input.append(vec_grade(list_grade_genre))
			test_attempt.append(vec_attempt(list_grade_genre))
			test_inverse.append(vec_inverse(list_grade_genre))

		real_input = vec_grade(real_input)

		#fourth, we evaluate the dataset properly
		ptr = 0
		overall_diff = 0.
		no_batches_new_test = int(len(test_input) / batchSize)
		for i in range(no_batches_new_test):
			batch_real = real_input[ptr:ptr+batchSize]
			batch_input = test_input[ptr:ptr+batchSize]
			batch_attempt = test_attempt[ptr:ptr+batchSize]
			batch_inverse = test_inverse[ptr:ptr+batchSize]

			ptr += batchSize
			matrix_test = sess.run(matrix_output, feed_dict = {matrix_input : batch_input, \
				attempted_input : batch_attempt, learning_rate : dumped_set[2]})

			matrix_test = vec_convert(matrix_test)

			zero_count = np.float32(np.count_nonzero(batch_inverse))

			matrix_square_diff = np.multiply(np.square(np.subtract(matrix_test, batch_real)), batch_inverse)
			total_square_diff = np.sum(matrix_square_diff)
			avg_diff_test = np.sqrt(np.divide(total_square_diff, zero_count))

			overall_diff += avg_diff_test / no_batches_new_test

		print("The Overall Difference in Customized Test Data is", overall_diff)

		#last, we show the example of the test data (the real one, the masked one and the converted result one)
		show_real = real_input[3425]
		show_input = np.reshape(test_input[3425], [1, num_genre])
		show_attempt = np.reshape(test_attempt[3425], [1, num_genre])
		show_inverse = np.reshape(test_inverse[3425], [1, num_genre])

		show_result = sess.run(matrix_output, feed_dict = {matrix_input : show_input, \
			attempted_input : show_attempt, learning_rate : dumped_set[2]})
		show_result = vec_convert(show_result)
		diff_result = np.multiply(np.abs(np.subtract(show_real, show_result)), show_inverse)

		count_percentage = np.count_nonzero(diff_result) / np.count_nonzero(show_inverse) * 100

		print("Showing one example of the test sample")
		print("The real test sample is\n", show_real)
		print("The masked test sample is\n", show_input)
		print("The converted test result is \n", show_result)
		print("The difference with real input is \n", diff_result)
		print("The percentage of nonzero elements is", count_percentage, "percent")
