# CS 487
# John Ossorgin
# Project 1

#libraries
import sys
import os.path
import pandas
import numpy

#my files
import perc1, ada1, sgd1

#perceptron
def runPerc(dataset):

	(trainingX,trainingY,testingX,testingY) = dataTransform(dataset)

	perc = perc1.perc1(eta=0.1, n_iter=10)
	
	perc.fit(trainingX,trainingY)
	
	predictions = perc.predict(testingX)

	error = calculateError(predictions,testingY)
	print("the RMSE is:\n", error)

#adaline
def runAda(dataset):

	(trainingX,trainingY,testingX,testingY) = dataTransform(dataset)

	ada = ada1.ada1(eta=0.01, n_iter=15, random_state=1)
	
	ada.fit(trainingX,trainingY)
	
	predictions = ada.predict(testingX)

	error = calculateError(predictions,testingY)
	print("the RMSE is:\n", error)


#SGD
def runSGD(dataset):

	(trainingX,trainingY,testingX,testingY) = dataTransform(dataset)

	sgd = sgd1.sgd1(eta=0.1, n_iter=10)
	
	sgd.fit(trainingX,trainingY)
	
	predictions = sgd.predict(testingX)

	error = calculateError(predictions,testingY)
	print("the error (RMSE) is:\n", error)


#function to split the dataset into training and test data
def dataTransform(dataset):

	#read in the dataset
	data = pandas.read_csv(dataset)
	data = data.fillna(0)
	#split into training and test, roughly 80/20
	#idea taken from https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
	split = numpy.random.rand(len(data)) < 0.8
	
	#pick training data
	training = data[split]
	
	#pick testing data
	testing = data[~split]

	#split training and testing into x and y
	trainingX = (numpy.array(training.iloc[:, :-1].values))
	trainingY = (numpy.array(training.iloc[:, :1].values))

	testingX = (numpy.array(testing.iloc[:, :-1].values))
	testingY = (numpy.array(testing.iloc[:, :1].values))
	
	trainingX = trainingX.astype(numpy.float)

	trainingY = trainingY.astype(numpy.float)

	testingX = testingX.astype(numpy.float)

	testingY = testingY.astype(numpy.float)

	# print("trianingX:\n",trainingX)
	# print("trianingY:\n",trainingY)
	# print("testingX:\n",testingX)
	# print("testingY:\n",testingY)
	
	#return the data
	return trainingX, trainingY, testingX, testingY

#find the errors using RMSE
def calculateError(predicted, actual):

	error = numpy.sqrt(((predicted - actual) ** 2).mean()) 

	return error

def main():
	#classifiers 
	options = ["perceptron","adaline","sgd"]

	#error checking - number of arguments
	if len(sys.argv) <= 1:
		print("no args provided")
		exit()

	#error checking - classifier name
	elif (sys.argv[1] not in options):
		print("invalid classifier: use \"perceptron\",\"adaline\", or \"sgd\"")
		exit()

	elif len(sys.argv) <= 2:
		print("no data set provided")
		exit()

	#classifier name
	classifier = sys.argv[1]

	#dataset path
	ds = sys.argv[2]

	#check for valid dataset
	if not os.path.isfile(ds):
		print("invalid dataset path")
		exit()

	print("Running " + classifier + " with dataset: " + ds)

	#perceptron
	if classifier == options[0]:
		runPerc(ds)
		
	#adaline
	elif classifier == options[1]:
		runAda(ds)		

	#sgd
	elif classifier == options[2]:
		runSGD(ds)

	#error
	else:
		print("error")
		exit()

#run the main method		
if __name__ == "__main__":
	main()