"""
This code is purposed to generate the fake test data
that will be used to have their result shown, just to know how
good the network actually is in evaluating randomized data
"""

import csv
import random
import numpy as np
import scipy.stats as sp

num_users = 80

#Step 1 : Extract the CSV file
#The objective here is to get the information about the topic
#and also the genre's code for one particular genre
info_genre = []

with open("info_qfactory_h1s1.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		info_genre.append(int(row[3]))

num_topics = max(info_genre)
num_genre = len(info_genre)

grade_user = []

#Step 2 : Determine the grade for each user's genre
#Here the assumption is that each user has completed all of the genres
#but with different kind of grade that will be obtained, in total of 4 types


#The second type of user will get the grade based on assumption on normality in topics
#The third type of user will get the grade just randomly

cat_user = []

#The first type of user will get the grade based on assumption on multiple normalities
#(just like the assumption made for establishing the train, val and test data)
def first_kind_grade():
	alpha = np.random.normal(loc = 0.0, scale = 1.0)
	cdf_user = sp.norm(0,1).cdf(alpha)
	if cdf_user > 0.9:
		user_category = 1
	elif 0.7 < cdf_user < 0.9:
		user_category = 2
	elif 0.3 < cdf_user < 0.7:
		user_category = 3
	elif 0.1 < cdf_user < 0.3:
		user_category = 4
	else:
		user_category = 5
	
	cat_user.append(user_category)
	user_topics = []
	for tmp in range(num_topics): #for each user, iterating over all topics to determine user's capabilities on certain topic
		ratio = np.random.uniform(low = 0.9, high = 1.1)
		new_alpha = ratio * alpha
		user_topics.append(np.random.normal(loc = new_alpha, scale = 0.5))

	grade_obtained = []
	for i in range(len(info_genre)):
		grade_apx = np.random.normal(loc = user_topics[info_genre[i] - 1], scale = 0.25)
		cdf_value = sp.norm(0,1).cdf(grade_apx)
		if cdf_value > 0.9:
			grade_assigned = "S"
		elif 0.7 < cdf_value < 0.9:
			grade_assigned = "A"
		elif 0.3 < cdf_value < 0.7:
			grade_assigned = "B"
		elif 0.1 < cdf_value < 0.3:
			grade_assigned = "C"
		else:
			grade_assigned = "D"

		grade_obtained.append(grade_assigned)

	grade_user.append(grade_obtained)

#The second type of user belongs to the user who has customized randomization, which is that
#user has really random rate of knowledge varying over topics on same grade
def second_kind_grade():
	cat_user.append(6)
	user_topics = []
	for tmp in range(num_topics): #for each user, iterating over all topics to determine user's capabilities on certain topic
		user_topics.append(np.random.normal(loc = 0.0, scale = 1.0))

	grade_obtained = []
	for i in range(len(info_genre)):
		grade_apx = np.random.normal(loc = user_topics[info_genre[i] - 1], scale = 0.25)
		cdf_value = sp.norm(0,1).cdf(grade_apx)
		if cdf_value > 0.9:
			grade_assigned = "S"
		elif 0.7 < cdf_value < 0.9:
			grade_assigned = "A"
		elif 0.3 < cdf_value < 0.7:
			grade_assigned = "B"
		elif 0.1 < cdf_value < 0.3:
			grade_assigned = "C"
		else:
			grade_assigned = "D"

		grade_obtained.append(grade_assigned)

	grade_user.append(grade_obtained)

#The third type of user may belong to the user who has the grades totally randomized
def third_kind_grade():
	cat_user.append(6)
	grade_obtained = []
	for i in range(len(info_genre)):
		grade_apx = np.random.normal(loc = 0.0, scale = 1.0)
		cdf_value = sp.norm(0,1).cdf(grade_apx)
		if cdf_value > 0.9:
			grade_assigned = "S"
		elif 0.7 < cdf_value < 0.9:
			grade_assigned = "A"
		elif 0.3 < cdf_value < 0.7:
			grade_assigned = "B"
		elif 0.1 < cdf_value < 0.3:
			grade_assigned = "C"
		else:
			grade_assigned = "D"

		grade_obtained.append(grade_assigned)

	grade_user.append(grade_obtained)

#The fourth type of user belongs to the uniformly distributed randomized grading system
def fourth_kind_grade():
	cat_user.append(6)
	grade_obtained = []
	for i in range(len(info_genre)):
		grade_apx = np.random.uniform(low = 0.0, high = 5.0)
		cdf_value = sp.uniform(0,5).cdf(grade_apx)
		if cdf_value > 0.8:
			grade_assigned = "S"
		elif 0.6 < cdf_value < 0.8:
			grade_assigned = "A"
		elif 0.4 < cdf_value < 0.6:
			grade_assigned = "B"
		elif 0.2 < cdf_value < 0.4:
			grade_assigned = "C"
		else:
			grade_assigned = "D"

		grade_obtained.append(grade_assigned)

	grade_user.append(grade_obtained)

for i in range(num_users):
	if i < num_users / 4:
		first_kind_grade()
	elif num_users / 4 <= i < num_users / 2:
		second_kind_grade()
	elif num_users / 2 <= i < 3 * num_users / 4:
		third_kind_grade()
	else:
		fourth_kind_grade()

#Now starting to writing up into the test data file
with open("test_grade_h1s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	list_elem = ["user_id", "category"]
	for i in range(num_genre):
		list_elem.append("Genre #" + str(i + 1))
	csv_writer.writerow(list_elem)

	for i in range(len(grade_user)):
		list_elem = ["User #" + str(i + 1)]
		list_elem = list_elem + [cat_user[i]] + grade_user[i]
		csv_writer.writerow(list_elem)
