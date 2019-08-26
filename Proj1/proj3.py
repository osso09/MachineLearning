# CS 487
# John Ossorgin
# Project 3

#libraries
import sys
import os.path
import pandas
import numpy
import sklearn
import time
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

#perceptron
def runPerc(dataset):

	print("\n\nrunning Perceptron")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(dataset)

	#begin timing
	startTime = time.time()

	#create perceptron object
	perc = Perceptron(max_iter=40, eta0=0.1, random_state=1)

	#fit the data
	perc.fit(trainingX,trainingY)

	#make predictions
	prediction = perc.predict(testingX)

	#show error
	print("Error score is: ",perc.score(testingX,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)

#svm - linear
def runSVMLinear(dataset):
	
	print("\n\nrunning SVM linear")
	
	#begin timing
	startTime = time.time()

	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(dataset)

	#begin timing
	startTime = time.time()

	#create svm object
	svm = SVC(kernel='linear')

	#fit the data
	svm.fit(trainingX,trainingY)

	#make predictions
	prediction = svm.predict(testingX)

	#show error
	print("Error score is: ",svm.score(testingX,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)

#svm - RBF
def runSVMRBF(dataset):
	
	print("\n\nrunning SVM with RBF kernel")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(dataset)

	#begin timing
	startTime = time.time()

	#create svm object
	svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)

	#fit the data
	svm.fit(trainingX,trainingY)

	#make predictions
	prediction = svm.predict(testingX)

	#show error
	print("Error score is: ",svm.score(testingX,testingY))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)

#decision tree
def runDecTree(dataset):
	
	print("\n\nrunning decision tree")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(dataset)

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

#knn
def runKNN(dataset):
	
	print("\n\nrunning KNN")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(dataset)

	#begin timing
	startTime = time.time()

	#create knn object
	knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

	#fit the data
	knn.fit(trainingX,trainingY)

	#make predictions
	prediction = knn.predict(testingX)

	#show error
	print("Error score is: ",knn.score(testingX,testingY))
	
	endTime = time.time()

	#print the runtime
	print("Runtime: ", endTime - startTime)
	
#logistic regression
def runLogReg(dataset):
	
	print("\n\nrunning logistic regression")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(dataset)

	#begin timing
	startTime = time.time()

	#create logistic regression object
	lr = LogisticRegression(C=100.0, random_state=1)

	#fit the data
	lr.fit(trainingX,trainingY)

	#make predictions
	prediction = lr.predict(testingX)

	#show error
	print("Error score is: ",lr.score(testingX,testingY))
	
	endTime = time.time()

	#print the runtime
	print("Runtime: ", endTime - startTime)

	
#function to split the dataset into training and test data
def dataTransform(dataset):


	#for running the digits dataset from scikit	
	if dataset == 'digits':
		digits = datasets.load_digits()
		x, y = digits.data, digits.target
	
	#for any other dataset entered by the user
	else:
		#read in the dataset
		data = pandas.read_csv(dataset, delimiter='\t')
		data = data.fillna(0)
	
		x = (numpy.array(data.iloc[:, 1:].values)).astype(numpy.float)
		y = (numpy.array(data.iloc[:, 0].values)).astype(numpy.float)

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
	options = ["perceptron","svm-linear","svm-rbf","decision", "knn", "logistic","all"]

	#error checking - number of arguments
	if len(sys.argv) <= 1:
		print("no args provided")
		exit()

	#error checking - classifier name
	elif (sys.argv[1] not in options):
		print("invalid classifier: use \"all\",\"perceptron\",\"svm-linear\",\"svm-rbf\",\"decision\",\"knn\", or \"logistic\"")
		exit()

	elif len(sys.argv) <= 2:
		print("no data set provided - use 'digits' for the scikit digits dataset")
		exit()

	#classifier name
	classifier = sys.argv[1]

	#dataset path
	ds = sys.argv[2]


	#check for valid dataset
	if not 'digits' and not os.path.isfile(ds):
		print("invalid dataset path")
		exit()


	#perceptron
	if classifier == options[0]:
		runPerc(ds)
		
	#svm - linear
	elif classifier == options[1]:
		runSVMLinear(ds)		

	#svm - rbf
	elif classifier == options[2]:
		runSVMRBF(ds)
	
	#decision tree
	elif classifier == options[3]:
		runDecTree(ds)

	#KNN
	elif classifier == options[4]:
		runKNN(ds)
	
	#Logistic regression
	elif classifier == options[5]:
		runLogReg(ds)

	#run all 
	elif classifier == options[6]:
		runPerc(ds)
		runSVMLinear(ds)
		runSVMRBF(ds)
		runDecTree(ds)
		runKNN(ds)
		runLogReg(ds)
		
	#error
	else:
		print("error")
		exit()

#run the main method		
if __name__ == "__main__":
	main()





