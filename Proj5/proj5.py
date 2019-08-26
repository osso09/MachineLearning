# CS 487
# John Ossorgin
# Project 5

#libraries
import sys
import os.path
import pandas
import numpy
import sklearn
import time

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets


def runKMeans(ds):
	
	print("\n\nrunning k-means")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#create kmeans object
	kmeans = KMeans(n_clusters=3, init='random', n_init=10, max_iter = 300, tol=1e-04, random_state=0)

	#fit the data
	kmeans.fit(trainingX,trainingY)

	#make predictions
	prediction = kmeans.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY)*len(prediction))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


def runLinkage(ds):
	
	print("\n\nrunning linkage")
	
	split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#create linkage object
	link = linkage(trainingY, method="complete", metric="euclidean")

	#fit the data
	link.fit(trainingX,trainingY)

	#make predictions
	prediction = link.predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY)*len(prediction))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


def runAgg(ds):
	
	print("\n\nrunning agglomerative")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#create linkage object
	agg = AgglomerativeClustering(n_clusters=3, linkage="complete", affinity="euclidean")

	#fit the data
	agg.fit(trainingX,trainingY)

	#make predictions
	prediction = agg.fit_predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY)*len(prediction)*len(prediction))

	endTime = time.time()

	print("Runtime in seconds: ", endTime - startTime)


def runDBScan(ds):
	
	print("\n\nrunning DBSCAN")
	
	#split the data
	(trainingX,testingX,trainingY,testingY) = dataTransform(ds)

	#begin timing
	startTime = time.time()

	#create linkage object
	dbs = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")

	#fit the data
	dbs.fit(trainingX,trainingY)

	#make predictions
	prediction = dbs.fit_predict(testingX)

	#show error
	print("Error score is: ",mean_squared_error(prediction,testingY)*len(prediction))

	endTime = time.time()

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
		data = pandas.read_csv(dataset)

		data = data.fillna(0)
		
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
	options = ["kmeans","linkage","agglomerative","dbscan","all"]

	#error checking - number of arguments
	if len(sys.argv) <= 1:
		print("no args provided")
		exit()

	#error checking - classifier name
	elif (sys.argv[1] not in options):
		print("invalid classifier: use \"kmeans\",\"linkage\",\"agglomerative\",\"dbscan\",or,\"all\"")
		exit()

	elif len(sys.argv) <= 2:
		print("no data set provided")
		exit()

	#classifier name
	classifier = sys.argv[1]

	#dataset path
	ds = sys.argv[2]

	#kmeans
	if classifier == options[0]:
		runKMeans(ds)
		
	#Linkage
	elif classifier == options[1]:
		runLinkage(ds)		

	#agglomerative
	elif classifier == options[2]:
		runAgg(ds)
	
	#DBSCAN
	elif classifier == options[3]:
		runDBScan(ds)

	#all
	elif classifier == options[4]:
		runKMeans(ds)
		runLinkage(ds)
		runAgg(ds)
		runDBScan(ds)
				
	#error
	else:
		print("error")
		exit()

#run the main method		
if __name__ == "__main__":
	main()





