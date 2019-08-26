# CS 487
# John Ossorgin
# Project 4
# Helper class for normal equation

import numpy

class normal:
	
	#constructor
	def __init__(self):
		self.w = None

	#fit function based on notes
	def fit(self, x, y):
		
		#append ones
		ones = numpy.ones((x.shape[0]))
		ones = ones[:, numpy.newaxis]
		newX = numpy.hstack((ones, x))
		
		#reshape
		self.w = numpy.zeros(x.shape[1])
		
		#incvert
		z = numpy.linalg.inv(numpy.dot(newX.T, newX))
		
		#get dot product
		self.w = numpy.dot(z, numpy.dot(newX.T, y))
		
	#predict 
	def predict(self, x):
		return numpy.dot(x, self.w[1])