# CS 487
# John Ossorgin
# Project 8

#libraries
import sys
import os.path
import pandas
import numpy
import sklearn
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

def runBase(ds):
	
	print("\n\nrunning baseline Decision Tree")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run logistic regression
	tree = DecisionTreeClassifier(criterion='entropy', random_state=1,max_depth=None)
	tree.fit(trainingX, trainingY)
	prediction = tree.predict(testingX)

	endTime = time.time()
	
	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)

def runRand(ds):
	
	print("\n\nrunning Random Forest")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1)
	forest.fit(trainingX,trainingY)
	prediction = forest.predict(testingX)
	endTime = time.time()
	
	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)


def runBag(ds):
	
	print("\n\nrunning Bagging")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()
	tree = DecisionTreeClassifier(criterion='entropy', random_state=1,max_depth=None)
	bag = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
	bag.fit(trainingX,trainingY)
	prediction = bag.predict(testingX)
	endTime = time.time()

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)


def runAda(ds):
	
	print("\n\nrunning AdaBoost")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run AdaBoost
	tree = DecisionTreeClassifier(criterion='entropy', random_state=1,max_depth=None)
	ada = AdaBoostClassifier(base_estimator=tree, n_estimators=100, learning_rate=0.1, random_state=1)
	ada.fit(trainingX, trainingY)
	prediction = ada.predict(testingX)

	endTime = time.time()

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)

#function to split the dataset into training and test data
def dataTransform(dataset):

	
	if "digits" in dataset:
		# read in the dataset
		data = datasets.load_digits()

		x = data.data
		y = data.target
	

	else:
		# read in the dataset
		data = pandas.read_csv(dataset)

		data = data.fillna(0)
		data = data.replace('?',0)
		x = (numpy.array(data.iloc[:, :5].values)).astype(numpy.float)
		y = (numpy.array(data.iloc[:, 5].values)).astype(numpy.float)
	
	#split the data
	trainingX,testingX,trainingY,testingY = train_test_split(x, y, test_size=0.3, random_state=1)
	

	#scale the data
	scaler = StandardScaler()
	
	scaler.fit(trainingX)

	trainingX = scaler.transform(trainingX)
	testingX = scaler.transform(testingX)
	
	#return the data
	return trainingX,testingX,trainingY,testingY


#main function to run classifier based on command line args
def main():
	
	#classifiers 
	options = ["Random","Bagging","AdaBoost","all"]

	#error checking - number of arguments
	if len(sys.argv) <= 1:
		print("no args provided")
		exit()

	#error checking - classifier name
	elif (sys.argv[1] not in options):
		print("invalid classifier: use \"Random\",\"Bagging\",\"AdaBoost\", or \"all\"")
		exit()

	elif len(sys.argv) <= 2:
		print("no data set provided")
		exit()

	#classifier name
	classifier = sys.argv[1]

	#dataset path
	ds = sys.argv[2]

	#run a baseline for unchnaged numbers
	runBase(ds)
	
	#Random Forest
	if classifier == options[0]:
		runRand(ds)
		
	#Bagging
	elif classifier == options[1]:
		runBag(ds)		

	#AdaBoost
	elif classifier == options[2]:
		runAda(ds)
	
	#all
	elif classifier == options[3]:
		runRand(ds)
		runBag(ds)
		runAda(ds)
				
	#error
	else:
		print("error")
		exit()

#run the main method		
if __name__ == "__main__":
	main()





