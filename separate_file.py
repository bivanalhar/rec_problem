"""
This file is intended to split the csv file of the QFactory Statistics 
into several csv files, each is unique based on semester identity
"""

import csv

appended_row_h1s1 = []
appended_row_h1s2 = []
appended_row_hcal = []
appended_row_hsta = []
appended_row_hsu1 = []
appended_row_hsu2 = []
appended_row_m1s1 = []
appended_row_m2s1 = []
appended_row_m3s1 = []

#Step No.1 : storing all of the H1S1 information into separate csv file
with open("info_qfactory.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")

	for row in csv_reader:
		if row[1] == "H1S1":
			appended_row_h1s1.append(row)
		elif row[1] == "H1S2":
			appended_row_h1s2.append(row)
		elif row[1] == "HCAL":
			appended_row_hcal.append(row)
		elif row[1] == "HSTA":
			appended_row_hsta.append(row)
		elif row[1] == "HSU1":
			appended_row_hsu1.append(row)
		elif row[1] == "HSU2":
			appended_row_hsu2.append(row)
		elif row[1] == "M1S1":
			appended_row_m1s1.append(row)
		elif row[1] == "M2S1":
			appended_row_m2s1.append(row)
		elif row[1] == "M3S1":
			appended_row_m3s1.append(row)
		else:
			print(row[1])

with open("info_qfactory_h1s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_h1s1:
		csv_writer.writerow(row)

with open("info_qfactory_h1s2.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_h1s2:
		csv_writer.writerow(row)

with open("info_qfactory_hcal.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_hcal:
		csv_writer.writerow(row)

with open("info_qfactory_hsta.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_hsta:
		csv_writer.writerow(row)

with open("info_qfactory_hsu1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_hsu1:
		csv_writer.writerow(row)

with open("info_qfactory_hsu2.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_hsu2:
		csv_writer.writerow(row)

with open("info_qfactory_m1s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_m1s1:
		csv_writer.writerow(row)

with open("info_qfactory_m2s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_m2s1:
		csv_writer.writerow(row)

with open("info_qfactory_m3s1.csv", mode = 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ",")

	for row in appended_row_m3s1:
		csv_writer.writerow(row)