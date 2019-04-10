import h5py as h5
import numpy as np
import sys
import time

'''
Script to create the validation data.
'''

class validationSet(object):

	def __init__(self, Q, dataPath, obsN, chanN):
		
		self.quarter = Q
		self.path = dataPath
		self.numberobs = obsN
		self.channels = chanN


	def loadValidation(self):

		dataSet = h5.File(self.path + 'data_validation', mode='w')

		train_data_shape = (self.numberobs, 554, 554, self.channels) # Corp so square img
		radar_data_shape = (self.numberobs, 2500)
		nObsUploaded = (1, 1)
		meanShape = (1, self.channels)
		stdShape = (1, self.channels)

		# Create template
		dataSet.create_dataset("inputData", train_data_shape, np.float32)
		dataSet.create_dataset("outputData", radar_data_shape, np.float32)
		dataSet.create_dataset("nObservations", nObsUploaded, np.float32)
		dataSet.create_dataset("mean", meanShape, np.float32)
		dataSet.create_dataset("std", stdShape, np.float32)

		# Sample n from each month
		subN = self.numberobs/len(self.quarter)

		# Index to store the load index
		count = 0

		# Loop for each month in the quarter
		for d in self.quarter:

			atmpath = self.path + 'data_' + d

			# Open the datafile
			atmDataset = h5.File(atmpath, 'r')

			# Fetch the number of obs in the dataset and then take a random subset
			obsIndataset = atmDataset['nObservations'][0][0]
			randomIndex = np.arange(obsIndataset)
			np.random.shuffle(randomIndex)

			# Upload subN from dataset d
			for i in range(subN):

				atmLoadInput = atmDataset['inputData'][randomIndex[i]]
				atmLoadOutput = atmDataset["outputData"][randomIndex[i]]

				dataSet["inputData"][count, ...] = atmLoadInput[None]
				dataSet["outputData"][count, ...] = atmLoadOutput[None]

				count += 1

				print 'Loading year {} and number of uploads {}'.format(d, count)

			atmDataset.close()

		dataSet["nObservations"][...] = np.array([count], dtype=np.float32)[None]
		dataSet.close()


def main():
	
	numberObs 		= 1000
	quarterMonths 	= ['201401', '201402', '201403']
	pathFiles 		= ''
	numberChannels 	= 15

	dataval = validationSet(quarterMonths, pathFiles, numberObs, numberChannels)
	dataval.loadValidation()




if __name__ == '__main__':
	main()

