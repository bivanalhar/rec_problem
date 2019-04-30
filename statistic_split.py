"""
The purpose of this code is to either
1. look at the statistics of the user data (capability based)
2. split the user data into training, validation and testing dataset
"""

import csv
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("lookup_mode", True, "if True, look at the statistics. Else, splitting the dataset")

if FLAGS.lookup_mode:
	#this is the mode for lookup at the statistics

	user_capable = [0, 0, 0, 0, 0]
	with open("user_grade_h1s1.csv") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ",")
		for row in csv_reader:
			if row[1] != 'category':
				user_capable[int(row[1]) - 1] += 1
	print(user_capable)

else:
	#this is the mode for splitting the file into training, validation and testing

	data_by_capability = [[], [], [], [], []]
	with open("user_grade_h1s1.csv") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ",")
		for row in csv_reader:
			if row[1] != 'category':
				data_by_capability[int(row[1]) - 1].append(row)

	train_split = [6500, 10000, 15000, 10000, 6500] # will be decided later
	val_split = [1673, 5004, 12595, 4957, 1771]
	test_split = [1672, 5005, 12595, 4957, 1771]

	with open("train_data.csv", mode = 'w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter = ',')
		for i in range(len(train_split)):
			for j in range(train_split[i]):
				csv_writer.writerow(data_by_capability[i][j])

	with open("val_data.csv", mode = "w") as csv_file:
		csv_writer = csv.writer(csv_file, delimiter = ',')
		for i in range(len(val_split)):
			for j in range(val_split[i]):
				csv_writer.writerow(data_by_capability[i][j + train_split[i]])

	with open("test_data.csv", mode = "w") as csv_file:
		csv_writer = csv.writer(csv_file, delimiter = ",")
		for i in range(len(test_split)):
			for j in range(test_split[i]):
				csv_writer.writerow(data_by_capability[i][j + train_split[i] + val_split[i]])