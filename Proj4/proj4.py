# CS 487
# John Ossorgin
# Project 4

#libraries
import sys
import os.path
import pandas
import numpy
import sklearn
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, RANSACRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#my helper class
from normal import normal


def runLinReg(ds):

	print("\n\nrunning Linear Regression")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run regressor
	linReg = LinearRegression()

	#fit the data
	linReg.fit(trainingX,trainingY)

	#make predictions
	prediction = linReg.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)

def runRANSAC(ds):

	print("\n\nrunning RANSAC")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run regressor
	ransac = RANSACRegressor(random_state=0)

	#fit the data
	ransac.fit(trainingX,trainingY)

	#make predictions
	prediction = ransac.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


def runRidge(ds):

	print("\n\nrunning Ridge")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run regressor
	ridge = Ridge(alpha=1.0)

	#fit the data
	ridge.fit(trainingX,trainingY)

	#make predictions
	prediction = ridge.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


def runLasso(ds):

	print("\n\nrunning Lasso")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run regressor
	lasso = Lasso(alpha=0.1)

	#fit the data
	lasso.fit(trainingX,trainingY)

	#make predictions
	prediction = lasso.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


#Non linear regression using svm with RBF kernel from project 3
def runNonLinear(ds):
	
	print("\n\nrunning decision tree")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#create decision tree object
	dtc = DecisionTreeClassifier()

	#fit the data
	dtc.fit(trainingX,trainingY)

	#make predictions
	prediction = dtc.predict(testingX)

	#show error
	print("Error score is: ",dtc.score(testingX,testingY))
	
	endTime = time.time()

	#print the runtime
	print("Runtime: ", endTime - startTime)

def runNorm(ds):
	
	print("\n\nrunning normal equation")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#create svm object
	norm = normal()

	#fit the data
	norm.fit(trainingX,trainingY)

	#make predictions
	prediction = norm.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


#function to split the dataset into training and test data
def dataTransform(dataset):

	
	if "housing" in dataset:
		# read in the dataset
		data = pandas.read_csv(dataset, delim_whitespace=True)

		# Defining the columns
		data.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

		# Separating x and y
		x = data.iloc[:, :-1].values
		y = data["MEDV"].values

	else:
		# read in the dataset
		data = pandas.read_csv(dataset)

		data = data.dropna(axis='columns')
		
		x = (numpy.array(data.iloc[:, 2:].values)).astype(numpy.float)
		y = (numpy.array(data.iloc[:, 1].values)).astype(numpy.float)
	
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
	options = ["LinearRegression","RANSAC","Ridge","Lasso","Normal","NonLinear","all"]

	#error checking - number of arguments
	if len(sys.argv) <= 1:
		print("no args provided")
		exit()

	#error checking - classifier name
	elif (sys.argv[1] not in options):
		print("invalid classifier: use \"LinearRegression\",\"RANSAC\",\"Ridge\",\"Lasso\",\"Normal\",\"NonLinear\",\"all\"")
		exit()

	elif len(sys.argv) <= 2:
		print("no data set provided")
		exit()

	#classifier name
	classifier = sys.argv[1]

	#dataset path
	ds = sys.argv[2]

	#Linear Regression
	if classifier == options[0]:
		runLinReg(ds)
		
	#RANSAC
	elif classifier == options[1]:
		runRANSAC(ds)		

	#Ridge
	elif classifier == options[2]:
		runRidge(ds)
	
	#Lasso
	elif classifier == options[3]:
		runLasso(ds)

	#normal equation
	elif classifier == options[4]:
		runNorm(ds)

	#NonLinear
	elif classifier == options[5]:
		runNonLinear(ds)
	
	#run all 
	elif classifier == options[6]:
		runLinReg(ds)
		runRANSAC(ds)
		runRidge(ds)
		runLasso(ds)
		# runNorm(ds)
		# runNonLinear(ds)

		
	#error
	else:
		print("error")
		exit()

#run the main method		
if __name__ == "__main__":
	main()





