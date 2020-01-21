import numpy as np
import csv

trainingFilePath = "~/Drive/Skynet/numerai_training_data.csv"
tournamentFilePath = "~/Drive/Skynet/numerai_tournament_data.csv"

tournamentFile = open(tournamentFilePath, 'w+')

def readTraining(splitforTesting = True):
	trainingFile = open(trainingFilePath, 'rb')
	trainingFeatures = []
	trainingLabels = []

	rownum = 0
	reader = csv.reader(trainingFile)
	for row in reader:
	    # Save header row.
	    if rownum == 0:
	        header = row
	    else:
	        features = row[:len(row) - 1]
	       	trainingFeatures.append(np.array(features).astype('float32'))
	       	trainingLabels.append(np.array([row[-1]]).astype('float32'))

	            
	    rownum += 1
	if splitforTesting:
		testingFeatures = trainingFeatures[len(trainingFeatures) - 10000:]
		testingLabels = trainingLabels[len(trainingLabels) - 10000:]

		trainingFeatures = trainingFeatures[:len(trainingFeatures) - 10000]
		testingLabels = testingLabels[:len(testingLabels) - 10000]
		return trainingFeatures, trainingLabels, testingFeatures, testingLabels

	else:
		return trainingFeatures, trainingLabels


def readTournament():
	tournamentFile = open(tournamentFilePath, 'w+')
	tournamentFeatures = []

	rownum = 0
	reader = csv.reader(tournamentFile)
	for row in reader:
	    # Save header row.
	    if rownum == 0:
	        header = row
	    else:
	        features = row[:]
	       	tournamentFeatures.append(np.array(features).astype('float32'))

	            
	    rownum += 1

	return tournamentFeatures

