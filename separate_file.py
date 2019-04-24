"""
This file is intended to split the csv file of the QFactory Statistics 
into several csv files, each is unique based on semester identity
"""

import csv

appended_row_h1s1 = []
appended_row_h1s2 = []

#First Step : storing all of the H1S1 information into separate csv file
with open("info_qfactory.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")

	for row in csv_reader:
		if row[0] == "H1S1":
			row.append(appended_row)

with open("info_qfactory_H1S1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row:
		csv_writer.writerow(row)