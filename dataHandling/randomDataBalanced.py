import h5py as h5
import numpy as np
import time
import sys
import os

class constructStochasticData(object):

	def __init__(self, pData, pSave, na):
		
		self.dataList = []
		self.dataListSize = []
		self.dataIndexList = []
		self.pathData = pData
		self.pathSave = pSave
		self.datasetName = na

	def meanFetch(self):

		files = os.listdir(self.pathData)

		for f in files:

			atmPath = self.pathData + f
			atmData = h5.File(atmPath, 'r')

			if files.index(f) == 0:
				Trainmean = atmData['mean'][0]
			else:
				Trainmean += atmData['mean'][0]

			atmData.close()

		self.meanData = Trainmean/len(files)


	def importData(self):
		# Method create a list with respective number of obs in each datafile
		files = os.listdir(self.pathData)

		for i, f in enumerate(files):
			self.dataList.append(h5.File(self.pathData + f, 'r'))


	def createIndex(self):

		self.totalObservations = 0

		frac15 = 0
		count = 0

		for j, data in enumerate(self.dataList):

			atmRain = np.array([])
			atmNonRain = np.array([])

			# Loop through all obs in dataset
			for i in range(data["nObservations"][0][0]):
				atmOut = data['outputData'][i]

				count += 1

				if np.mean(atmOut) > 0.15:
					atmRain = np.append(atmRain, i)
					frac15 += 1

				else:
					atmNonRain = np.append(atmNonRain, i)

				sys.stdout.write('\rIter: {}'.format(count))
				sys.stdout.flush()

			np.random.shuffle(atmNonRain)
			np.random.shuffle(atmRain)

			if len(atmNonRain) > len(atmRain):

				atmNonRain = atmNonRain[:len(atmRain)]
				balancedIndex = np.append(atmNonRain, atmRain)
				np.random.shuffle(balancedIndex)

			else:

				atmRain = atmRain[:len(atmNonRain)]
				balancedIndex = np.append(atmNonRain, atmRain)
				np.random.shuffle(balancedIndex)

			self.dataIndexList.append(balancedIndex)
			self.totalObservations += balancedIndex.size


		print "\n"
		print "Share: {}/{}".format(frac15, count)
		print "Total observation is {}".format(self.totalObservations)

	def createhdf5file(self, size, name):

		dataSet = h5.File(self.pathSave  + 'data_' + self.datasetName + '_' + name, mode='w')

		train_data_shape = (size, 15, 554, 554)
		radar_data_shape = (size, 2500)
		nObsUploaded = (1, 1)
		meanShape = (1, 15)

		# Create template
		dataSet.create_dataset("inputData", train_data_shape, np.float32)
		dataSet.create_dataset("outputData", radar_data_shape, np.float32)
		dataSet.create_dataset("nObservations", nObsUploaded, np.float32)
		dataSet.create_dataset("mean", meanShape, np.float32)

		return dataSet


	def createDataset(self):
		'''
		Script works by creating a index over the dataset for each source data and taking a subset so we have 50% rain and 50% non rain.
		Then this list is shuffled around so each index list is randomized. 
		Then we loop over these indices and grab the first element and store it in a hdf5 file.
		Then we remove this element from the list and continue. If the list is empty we continue 
		and ignore this list. This loop continue until we have uploaded all the obs.
		'''
		print "Start randomize collection" 
		self.meanFetch()
		self.importData()
		self.createIndex()

		nSize = 1000
		dataHD = None

		count = 0

		timer = time.time()
		while True:

			for i, d in enumerate(self.dataList):

				if len(self.dataIndexList[i]) == 0:
					continue # If the index is empty, ignore this datafile

				if count % nSize == 0:

					if not dataHD == None:
						dataHD["nObservations"][...] = np.array([nobsThisFile], dtype=np.float32)[None]
						dataHD["mean"][...] = self.meanData[None]
						dataHD.close()

					dataHD = self.createhdf5file(nSize, str(count))
					nobsThisFile = 0

				atmIndex = self.dataIndexList[i][0]
				atmInput = d['inputData'][atmIndex] - self.meanData # REMOVE MEAN HERE, data fetcher changed
				atmInput = d['inputData'][atmIndex].T # Fix so shape is 15,554,554
				atmOutput = d['outputData'][atmIndex]

				dataHD['inputData'][nobsThisFile, ...] = atmInput[None]
				dataHD['outputData'][nobsThisFile, ...] = atmOutput[None]
				nobsThisFile += 1

				# Remove index
				self.dataIndexList[i] = np.delete(self.dataIndexList[i], 0) # From this list remove the first element

				sys.stdout.write('\rComplete upload:{}% Total sec: {}'.format(round((count + 1)/float(self.totalObservations) * 100, 3), round(time.time() - timer), 2))
				sys.stdout.flush()
				count += 1

			if count == self.totalObservations:

				# Add the number of uploaded 
				dataHD["nObservations"][...] = np.array([nobsThisFile], dtype=np.float32)[None]
				dataHD["mean"][...] = self.meanData[None]
				dataHD.close()
				print "Finish!"
				break			

	def destructor(self):

		for d in self.dataList:
			d.close()

def main():

	nameData 	= ""
	pathSource 	= ""
	pathSave 	= ""

	data = constructStochasticData(pathSource, pathSave, nameData)
	data.createDataset()
	data.destructor()

if __name__ == '__main__':
	main()