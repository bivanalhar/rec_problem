"""
This code is purposed to generate the fake dataset that is 
intended to be used for training the Collaborative Filtering
Algorithm. 
"""

import csv
import random

def fake_sequence(nbr_genres):
	#to generate the arrays of 0 and 1 with length nbr_genres
	#such that user has done that genre if the value is 1 and
	#user has not done that genre if the value is 0

	array_result = [0 for i in range(nbr_genres)]

	#number_one and number_zero denotes the number of elements 
	#inside that array_result with the value 1 and 0, respectively
	percent_one = random.randint(30, 90)
	number_one = floor(percent_one * nbr_genres / 100)

	#generating the list of randomized integer
	array_one = random.sample(range(nbr_genres), number_one)

	#setting the value of 0 and 1 appropriately
	for index in array_one:
		array_result[index] = 1

	return array_result

def fake_grade(array_result):
	#now we are about to produce the grade for each genre that
	#user has attempted (based on the result on the fake_sequence)

	array_grade = [None for i in range(len(array_result))]
	for i in range(len(array_result)):
		if array_result[i] == 1:
			#TODO : estimating the grade that this user may obtain based on the genre's questions
			pass

		else:
			array_grade[i] = "..."

	return array_grade