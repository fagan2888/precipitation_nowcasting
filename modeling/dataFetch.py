import numpy as np
import h5py
import os

def validationFetch(path, mean):
	data = h5py.File(path, 'r')
	x, y = (data['inputData'] - mean)[:,].transpose(0,3,2,1), np.array(data['outputData'])
	data.close()

	return x, y

def meanFetch(p):

	files = os.listdir(p)

	for f in files:

		atmPath = p + f
		atmData = h5py.File(atmPath, 'r')

		if files.index(f) == 0:
			Trainmean = atmData['mean'][0]
		else:
			Trainmean += atmData['mean'][0]

		atmData.close()

	return Trainmean/len(files)

class handleData(object):

	def __init__(self, bs, path, m):

		self.DataPath = path
		self.Dataset = h5py.File(self.DataPath, 'r')

		self.batchSize = bs

		# Mean of the train images
		self.Trainmean = m

		# N obs in the dataset
		self.nObs = self.Dataset['nObservations'][0][0] - self.Dataset['nObservations'][0][0] % self.batchSize

		# Calculate the batch indices
		self.index_obs = np.arange(self.nObs)

		# Create the array index
		self.batches_n = np.split(self.index_obs, self.nObs/self.batchSize)

	def getNBatch(self):
		return len(self.batches_n)

	def getBatch(self, idx):

		start = int(self.batches_n[idx][0])
		end = int(self.batches_n[idx][-1])

		return (self.Dataset['inputData'][start:end, ...] - self.Trainmean)[:,].transpose(0,3,2,1), self.Dataset['outputData'][start:end, ...]

	def closeData(self):
		self.Dataset.close()