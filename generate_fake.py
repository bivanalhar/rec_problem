"""
This code is purposed to generate the fake dataset that is 
intended to be used for training the Collaborative Filtering
Algorithm. 
"""

import csv
import random

num_users = 1000000 #number of fake users that will be created

#Step 1 : Extract the CSV file (the name of the file is info_qfactory_*.csv)
#Objective = to get the information about possibilities of students to get
#S, A, B, C, D, respectively
info_genre = []
with open("info_qfactory_h1s1.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		level_row = row[8:13]
		level_row = [int(number) + 2 for number in level_row]

		sum_level = sum(level_row)
		level_row = [int(number / sum_level * 50) for number in level_row]

		info_genre.append(level_row)

num_genre = len(info_genre)

#Step 2 : deciding on the number of random percentage of genre that user has 
#done for the particular semester
random_user = [random.choice(range(41)) for i in range(num_users)]
for i in range(len(random_user)):
	random_user[i] += 30

count_done_user = [int(percent * num_genre / 100) for percent in random_user]

grade_done_user = []

for i in range(len(count_done_user)):
	len_genre = list(range(num_genre))
	random.shuffle(len_genre)
	done_user = len_genre[:count_done_user[i]]
	done_user.sort()

	grade_done_user.append(done_user)

grade_user = [["N" for i in range(num_genre)] for j in range(num_users)]

#Step 3 : Deciding on the randomized grade for each of the user
for i in range(len(grade_done_user)):
	for j in range(len(grade_done_user[i])):
		list_5 = info_genre[grade_done_user[i][j]]
		grade_5 = ["S"] * list_5[0] + ["A"] * list_5[1] + ["B"] * list_5[2] \
			+ ["C"] * list_5[3] + ["D"] * list_5[4]
		random.shuffle(grade_5)
		grade_user[i][grade_done_user[i][j]] = grade_5[0]

#Step 4 : Rewriting into the csv file about the grade achieved
with open("user_grade_h1s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	list_elem = ["user_id"]
	for i in range(num_genre):
		list_elem.append("Genre #" + str(i + 1))
	csv_writer.writerow(list_elem)

	for i in range(len(grade_user)):
		list_elem = ["User #" + str(i + 1)]
		list_elem = list_elem + grade_user[i]
		csv_writer.writerow(list_elem)
