# CS 487
# John Ossorgin
# Project 6

#libraries
import sys
import os.path
import pandas
import numpy
import sklearn
import time

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA as KPCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

def runBase(ds):
	
	print("\n\nrunning baseline with no dimensionality reduction")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#run logistic regression
	lr = LogisticRegression(C=100.0, random_state=1)
	lr.fit(trainingX, trainingY)
	prediction = lr.predict(testingX)

	endTime = time.time()
	
	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)

def runPCA(ds):
	
	print("\n\nrunning PCA")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#reduce the dimensions
	pca = PCA(n_components=None)
	trainingX = pca.fit_transform(trainingX) 
	testingX = pca.transform(testingX)
	#run logistic regression
	lr = LogisticRegression(C=100.0, random_state=1)
	lr.fit(trainingX, trainingY)
	prediction = lr.predict(testingX)

	endTime = time.time()
	
	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)


def runLDA(ds):
	
	print("\n\nrunning LDA")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#reduce the dimensions
	lda = LDA(n_components=2)
	trainingX = lda.fit_transform(trainingX, trainingY) 
	testingX = lda.transform(testingX)

	#run logistic regression
	lr = LogisticRegression(C=100.0, random_state=1)
	lr.fit(trainingX, trainingY)
	prediction = lr.predict(testingX)

	endTime = time.time()

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	

	print("Runtime in seconds: ", endTime - startTime)


def runKPCA(ds):
	
	print("\n\nrunning Kernel PCA")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#reduce the dimensions
	kpca = KPCA(n_components=2, kernel='rbf', gamma=15)
	trainingX = kpca.fit_transform(trainingX) 
	testingX = kpca.transform(testingX) 

	#run logistic regression
	lr = LogisticRegression(C=100.0, random_state=1)
	lr.fit(trainingX, trainingY)
	prediction = lr.predict(testingX)

	endTime = time.time()

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY))

	print("Runtime in seconds: ", endTime - startTime)

#function to split the dataset into training and test data
def dataTransform(dataset):

	
	if "iris" in dataset:
		# read in the dataset
		data = datasets.load_iris()
		
		x = data.data[:, :2]  # we only take the first two features.
		y = data.target
	

	else:
		# read in the dataset
		data = datasets.load_digits()

		x = data.data
		y = data.target
	
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
	options = ["PCA","LDA","KernelPCA","all"]

	#error checking - number of arguments
	if len(sys.argv) <= 1:
		print("no args provided")
		exit()

	#error checking - classifier name
	elif (sys.argv[1] not in options):
		print("invalid classifier: use \"PCA\",\"LDA\",\"KernelPCA\", or \"all\"")
		exit()

	elif len(sys.argv) <= 2:
		print("no data set provided")
		exit()

	#classifier name
	classifier = sys.argv[1]

	#dataset path
	ds = sys.argv[2]

	runBase(ds)
	
	#Principal Component analysis
	if classifier == options[0]:
		runPCA(ds)
		
	#Linear Discriminant
	elif classifier == options[1]:
		runLDA(ds)		

	#kernel pca
	elif classifier == options[2]:
		runKPCA(ds)
	
	#all
	elif classifier == options[3]:
		runPCA(ds)
		runLDA(ds)
		runKPCA(ds)
				
	#error
	else:
		print("error")
		exit()

#run the main method		
if __name__ == "__main__":
	main()





