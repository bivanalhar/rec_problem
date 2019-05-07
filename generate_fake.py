"""
This code is purposed to generate the fake dataset that is 
intended to be used for training the Collaborative Filtering
Algorithm. 
"""

import csv
import random
import numpy as np
import scipy.stats as sp

num_users_1 = 50000 #number of fake users that will be created
num_users_2 = 25000
num_users_3 = 25000

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

#Step 2 : deciding on the number of random percentage of genre that user has 
#done for the particular semester
random_user_1 = [random.choice(range(21)) for i in range(num_users_1)]
random_user_2 = [random.choice(range(41)) for i in range(num_users_2)]
random_user_3 = [random.choice(range(11)) for i in range(num_users_3)]

for i in range(len(random_user_1)):
	random_user_1[i] += 10
for i in range(len(random_user_2)):
	random_user_2[i] += 31
for i in range(len(random_user_3)):
	random_user_3[i] += 71

num_users = num_users_1 + num_users_2 + num_users_3

random_user = random_user_1 + random_user_2 + random_user_3
random.shuffle(random_user)
count_done_user = [int(percent * num_genre / 100) for percent in random_user]

grade_done_user = []

for i in range(len(count_done_user)):
	len_genre = list(range(num_genre))
	random.shuffle(len_genre)
	done_user = len_genre[:count_done_user[i]]
	done_user.sort()

	#Storing the information about user's topic and genre that has been finished
	topic_done = [info_genre[index] - 1 for index in done_user]
	grade_done_user.append([topic_done, done_user])

grade_user = [["N" for i in range(num_genre)] for j in range(num_users)]

#Step 3 : Deciding on the randomized grade for each of the user, step-by-step
"""
The detailed sub-steps taken for this step is as follows:
1. First, determine the user's numerized capability (assume it belongs to N(0, 1))
2. Then, we determine the user's numerized topic-specific capability
	(assume alpha as result of 1, then beta ~ N([0.9~1.1]*alpha, 0.5))
3. Lastly, we determine the randomized grade of each genre, based on topic capability
	(we will use gamma ~ N(beta, 1/4) for determining the final grade)
"""
cat_user = []
for i in range(len(grade_done_user)): #iterating over all 100,000 users
	if i % 2000 == 1999:
		print("Now designing User #" + str(i + 1))

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
	
	list_genre_done = grade_done_user[i][0]
	idx_genre_done = grade_done_user[i][1]
		
	for l in range(len(list_genre_done)):
		grade_apx = np.random.normal(loc = user_topics[list_genre_done[l]], scale = 0.25)
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

		grade_user[i][idx_genre_done[l]] = grade_assigned

#Step 4 : Rewriting into the csv file about the grade achieved
with open("user_grade_h1s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	list_elem = ["user_id", "category"]
	for i in range(num_genre):
		list_elem.append("Genre #" + str(i + 1))
	csv_writer.writerow(list_elem)

	for i in range(len(grade_user)):
		list_elem = ["User #" + str(i + 1)]
		list_elem = list_elem + [cat_user[i]] + grade_user[i]
		csv_writer.writerow(list_elem)
