import h5py as h5
import numpy as np
from netCDF4 import Dataset
import sys
import time

class indexVectorVar(object):
	'''
	Class to store an index, used for the variables and the lags
	'''
	def __init__(self):
		self.VectorIndex = np.array([])

	def setidx(self, idx):
		self.VectorIndex = np.append(self.VectorIndex, idx)

	def getidx(self, at):
		return int(self.VectorIndex[at])

	def getlen(self):
		return len(VectorIndex)

class hdf5Data(object):
	'''
	Class to create HDF5 objects with index vectors
	'''
	def __init__(self, pathFile, name, lagVar):
		self.data = Dataset(pathFile, 'r')
		self.timesstamp = self.data['timestamps']
		self.nameVar = name

		# Add one so we can make the loop
		self.nlags = lagVar + 1

		# If radar, this is the forecast index i.e. t + forecast min
		if name == 'radar':
			self.radarForecastIdx = indexVectorVar()

		# Indexlist to create dataset
		self.IndexList = []

		for i in range(self.nlags):
			self.IndexList.append(indexVectorVar())

	def setIndex(self, lag, idx):
		self.IndexList[lag].setidx(idx)

	def getIndex(self, lag, at):
		return self.IndexList[lag].getidx(at)

	def getIndexVectorLen(self):
		self.IndexList[0].getlen()

	def getTimestampIdx(self, idx):
		if idx == -1337:
			res = -1337
		else:
			res = self.timesstamp[idx]
		return res

	def setRadarForceastIdx(self, idx):
		self.radarForecastIdx.setidx(idx)

	def getRadarForecastIdx(self, idx):
		return self.radarForecastIdx.getidx(idx)

	def getData(self, lag, idx):
		'''
		Method to fetch data given an index of the indexvector
		'''
		dataIndex = self.IndexList[lag].getidx(idx)
		return self.data[self.nameVar][dataIndex]

	def getRadarForecastData(self, idx):
		dataIndex = self.radarForecastIdx.getidx(idx)
		return self.data[self.nameVar][dataIndex]

	def close(self):
		self.data.close()

class periodData(object):
	'''
	Class to contain all the datafiles hdf5data for each variable
	'''
	def __init__(self, ECvar, MSvar, DATE, nlag, fh, obsNload):
		self.ECVarList = ECvar
		self.MSVarList = MSvar
		self.currentMonthSuffix = '_' + DATE + '.nc'
		self.forecastMin = fh
		self.nIndex = 0
		self.theDate = DATE
		self.totalUploadObs = obsNload

		# Empty arrays to store the HDF5 files
		self.EC_List = []
		self.MS_List = []
		self.RA_List = []

		# Lag dic
		self.LagDic = nlag

		# Dell path
		self.globalPathEC = ''
		self.globalPathMS = ''
		self.globalPathRD = ''

	def loadData(self):

		# The data that is loaded is based on the variables in the ECVarList and MSVarList

		# EC data
		for i in range(len(self.ECVarList)):
			atmPath = self.globalPathEC + 'ec_' + self.ECVarList[i] + self.currentMonthSuffix
			self.EC_List.append(hdf5Data(atmPath, self.ECVarList[i], self.LagDic['EC']))

		# Mesan data
		for i in range(len(self.MSVarList)):
			atmPath = self.globalPathMS + 'mesan_' + self.MSVarList[i] + self.currentMonthSuffix
			self.MS_List.append(hdf5Data(atmPath, self.MSVarList[i], self.LagDic['MS']))

		# Radar data
		self.RA_List.append(hdf5Data(self.globalPathRD + 'radar' + self.currentMonthSuffix, 'radar', self.LagDic['RA']))
		print 'HDF5 datasets opened'

	def findStartFinish(self, listData, Index, RadarTime):
		'''
		Method which compare the starttime of each variable and returns 
		highest start minute. Note that the variables always start at the 
		same time except for when the EC variable start (rare case), check EC var

		Also finds the last variable with lowest minute and subtact 180. By 
		doing so we will always have lagged data
		Arg: 
			listData: List of the time variables
			Index: boolean if we want radar index returned

		Return:
			List of two integers, first points to the start minute/index and
			second to the finish minute/index of the radardata.

		'''
		highList = []
		lowList = []

		for l in listData:
			highList.append(l[0])
			lowList.append(l[len(l) - 1])

		highestMinute = highList[np.argmax(highList)]
		lowestMinute = lowList[np.argmin(lowList)]

		# Added so we count the number of lags, this is needed so we have data availble to the lags
		msLag = self.LagDic['MS'] * 60
		raLag = self.LagDic['RA'] * 15

		if msLag > raLag:
			lagSubTime = msLag
		else:
			lagSubTime = raLag 

		# 180 is a fixed number so we assure we have data
		if Index:
			result = [self.findIndex(RadarTime, highestMinute + lagSubTime + 180), self.findIndex(RadarTime, lowestMinute - 180)]
		else:
			result = [highestMinute + 180 + lagSubTime, lowestMinute - 180]

		return result

	def createStartFinishIndex(self):
		'''
		Method which fetches all the variables timestamps and
		finds the min, max starting point

		'''
		listMaxMin = []

		RadarTimestamp = self.RA_List[0].timesstamp

		listMaxMin.append(RadarTimestamp)

		for f in self.EC_List:
			listMaxMin.append(f.timesstamp)

		for f in self.MS_List:
			listMaxMin.append(f.timesstamp)

		return self.findStartFinish(listMaxMin, True, RadarTimestamp)

	def findIndex(self, vector, minute):
		'''
		Method to fetch the index of some minute. I.e. 
		we want to find the index where the minute x is in the vector
		Arg:
			Vector: List to search over
			Minute: Minute to find the index
		Return:
			res: Index or a number which indicate that that this
			minute does not exists i.e. NA
		'''
		res = np.argwhere(np.array(vector) == minute)

		if not res and not res == 0:
			res = -1337
		else:
			res = res[0][0]
		return res

	def findNearLowVar(self, Var, minute, timebase):
		'''
		Method which return the lowest correct index, that is
		the lowest nearest minute to Var. 
		Here we assume that minute > Var for now (Fix later on)
		
		Example: We have radar data at 750 min. We want to find the
		lowest nearest EC data which is recorded at 720 min. This function
		will then return the index of 720 min in EC data vector

		Arg:
			Var:        The data where to find the nearest e.g. EC (timestamp vector)
			minute:     The current min e.g. radar data at 735
			timebase:   180 for EC data and 60 for mesan

		Return: Index of where the lowest and nearest observation of
				the specified variable is
		'''
		return self.findIndex(Var, minute - minute%timebase)

	def createDataIndex(self):

		# Start and end point of the radardata (Forcast serie)
		startFinish = self.createStartFinishIndex()

		# Radartime
		radarTime = self.RA_List[0].timesstamp

		count = 0

		# Start index construction from the constructed start and finish
		for i in range(len(radarTime))[startFinish[0]:startFinish[1]]:

			# For each variable list and each sub variable, create the correct index for the jth lag
			for var, t in zip([self.EC_List, self.MS_List, self.RA_List], [180, 60, 15]):

				for l in var:

					for j in range(l.nlags):

						# Add the forecasted radar index i.e. t + forecast
						if l.nameVar == 'radar' and j == 0:
							l.setRadarForceastIdx(i)

						# Here we find the correct index. In the current radartime (radarTime[i])
						# we want to go back fh min (forecast min) back and if we need lags variable j != 0
						# so we go more minute back in time. Second argument is to find the nearest obs

						# Here we will work with the EC type variables
						if t == 180:

							atmRadarTime = radarTime[i]

							# Fix so we get the closes EC var to the radarForecast
							atmIdx = self.findNearLowVar(l.timesstamp, atmRadarTime, t)

							if not atmIdx == -1337:
								# In the current "interval" get the EC timestamp nearest to RA
								minECDiff = abs(atmRadarTime - l.timesstamp[atmIdx])
								maxECDiff = abs(atmRadarTime - l.timesstamp[atmIdx + 1])

								if minECDiff > maxECDiff:
									atmIdx = atmIdx + 1

							l.setIndex(j, atmIdx)

						else:
							atmIdx = self.findNearLowVar(l.timesstamp, radarTime[i] - self.forecastMin - j * t, t)
							l.setIndex(j, atmIdx)

			if not self.totalUploadObs == None:
				if count == self.totalUploadObs:
					break

			count += 1

		self.nIndex = count

	def checkMissing(self, ith):
		'''
		Method to check if the ith index in all variables contain -1337
		Return boolean type, True == -1337 in the index "row"
		'''
		checkarr = np.array([])

		for f in self.EC_List:              
			idx = f.getIndex(0, ith)
			checkarr = np.append(checkarr, idx)

		for f in self.MS_List:
			for l in range(self.LagDic['MS'] + 1):
				idx = f.getIndex(l, ith)
				checkarr = np.append(checkarr, idx)

		for f in self.RA_List:
			for l in range(self.LagDic['RA'] + 1):
				idx = f.getIndex(l, ith)
				checkarr = np.append(checkarr, idx)
				idx = f.getRadarForecastIdx(ith)
				checkarr = np.append(checkarr, idx)

		return np.any(checkarr == -1337)

	def printIndex(self, printOut = True):
		'''
		Method to print the index converted to minutes to verify they are correct
		Could be print or written to a txt file
		'''
		if not printOut:
			txtfile = open('dataIndexCheck.txt', 'w')

		for i in range(self.nIndex):

			atmStr = ''

			for l in self.EC_List:

				# Assume no more than one lagged EC variable

				atmStr = '{}:{} '.format(str(l.nameVar), l.getTimestampIdx(l.getIndex(0, i))) + atmStr

			for l in self.MS_List:

				# Loop lags
				for lag in range(self.LagDic['MS'] + 1):

					atmStr = '{}Lag{}:{} '.format(str(l.nameVar), lag, l.getTimestampIdx(l.getIndex(lag, i))) + atmStr

			# Redundant loop since this list contain always only one element
			for l in self.RA_List:

				# Loop and print lagged
				for lag in range(self.LagDic['RA'] + 1):

					atmStr = '{}Lag{}:{} '.format(str(l.nameVar), lag, l.getTimestampIdx(l.getIndex(lag, i))) + atmStr

				# Add t + forecast radar minute
				atmStr = '{}Future:{} '.format(str(l.nameVar), l.getTimestampIdx(l.getRadarForecastIdx(i))) + atmStr


			if printOut:
				print atmStr + 'Missing: {}'.format(self.checkMissing(i))
			else:
				txtfile.write(atmStr + '\n')

		if not printOut:
			txtfile.close()

	def checkMode(self, mat, number, propotionCut):
		'''
		Method to check if mode of a number is over x % in dataset
		then return boolean
		'''
		nNumbers = mat[mat == number]
		prop = nNumbers.size/float(mat.size)

		return prop < propotionCut

	def checkMS(self, mat, number):
		'''
		Method to check if number do NOT exists in mat
		Returns boolean type 
		'''
		return not np.any(mat[mat == number])

	def createHDF5file(self, path, fix):

		totLag = self.LagDic['RA'] + self.LagDic['MS'] * len(self.MS_List) + self.LagDic['EC'] * len(self.EC_List) 
		totVar = len(self.EC_List) + len(self.MS_List) + len(self.RA_List)

		# Number of channels
		nChannels = totVar + totLag

		# Dataset shape
		train_data_shape = (self.nIndex, 554, 623, nChannels)
		if fix:
			train_data_shape = (self.nIndex, 554, 554, nChannels) # Corp so square img
		radar_data_shape = (self.nIndex, 2500)
		nObsUploaded = (1, 1)
		meanShape = (1, nChannels)
		stdShape = (1, nChannels)

		# Create hdf5 file
		dataSet = h5.File(path + 'data_' + self.theDate, mode='w')

		# Create template
		dataSet.create_dataset("inputData", train_data_shape, np.float32)
		dataSet.create_dataset("outputData", radar_data_shape, np.float32)

		# Create timestamp vector of the output. I.e. at the ith index we will have the corresponing
		# future radar image. Stored to check the dataset
		dataSet.create_dataset("outputDataTimestamp", (self.nIndex, 1), np.float32)

		# How many obs we upload
		dataSet.create_dataset("nObservations", nObsUploaded, np.float32)

		# Mean
		dataSet.create_dataset("mean", meanShape, np.float32)

		# Sd
		dataSet.create_dataset("std", stdShape, np.float32)

		# Initialize mean and std array
		meanArr = np.zeros(nChannels, dtype = np.float32) 
		stdArr = np.zeros(nChannels, dtype = np.float32)

		return dataSet, meanArr, stdArr

	def loadHDF5(self, path):

		# Boolean to indicate if we should fix the dim so its square matrix
		fixdim = True

		# Forecast radar image subset dim
		subDim = [218, 268, 385, 435]

		if fixdim:
			subDim = [218, 268, 351, 401] # fix forecast

		# Threshold to rain or not
		thresholdRain = 87

		# Radar whitespace fraction to ignore observation and which number (pixel)
		propWhite = 0.3
		pixelVal = 255

		# NA for mesan number representation
		naMesan = -999999

		# Create HDF5 file and initialize mean and std arrays
		dataFile, meanArr, stdArr = self.createHDF5file(path, fixdim)

		# Counter of how many we actually upload
		count = 0

		# Timer
		timer = time.time()

		for i in range(self.nIndex):

			# Flag to include obs or not
			includeObservation = True

			try:
				
				# Check if we have missing i.e. some index == -1337
				if not self.checkMissing(i):

					# Radar data (only one iter)
					for r in self.RA_List:

						for l in range(self.LagDic['RA'] + 1):

							if l == 0:

								dataTensor = r.getData(l, i)
								includeObservation = self.checkMode(dataTensor, pixelVal, propWhite)
								meanArr[0] += np.mean(dataTensor)
								stdArr[0] += np.std(dataTensor)

							else:

								if includeObservation:
									# Here we can take the mean and fix NA etc for radar
									atmTensor = r.getData(l, i)
									includeObservation = self.checkMode(atmTensor, pixelVal, propWhite)
									dataTensor = np.dstack((dataTensor, atmTensor))
									# Use period before so we dont need to recalculate, lags here
									meanArr[dataTensor.shape[2] - 1] = meanArr[dataTensor.shape[2] - 2]
									stdArr[dataTensor.shape[2] - 1] = stdArr[dataTensor.shape[2] - 2] 

						# Future radardata, reshape to vector and make binary at threshold.
						atmRadarVector = r.getRadarForecastData(i)

						# Subset so we only have over sthlm area and flatten i.e. vectorize
						if fixdim:
							atmRadarVector = atmRadarVector[:,34:588][subDim[0]:subDim[1], subDim[2]:subDim[3]].flatten()
						else:
							atmRadarVector = atmRadarVector[subDim[0]:subDim[1], subDim[2]:subDim[3]].flatten()

						# Make binary at the thereshold
						atmRadarVector = np.where(atmRadarVector > thresholdRain, 1, 0)


						# Get the timestamp of the forecast area
						forcastTimestampIndex = r.getRadarForecastIdx(i)
						forcastTimestamp = r.getTimestampIdx(forcastTimestampIndex)


					for e in self.EC_List:

						if includeObservation:

							atmTensor = e.getData(0, i)
							dataTensor = np.dstack((dataTensor, atmTensor))
							meanArr[dataTensor.shape[2] - 1] += np.mean(atmTensor)
							stdArr[dataTensor.shape[2] - 1] += np.std(atmTensor)

					for m in self.MS_List:

						for l in range(self.LagDic['MS'] + 1):

							if l == 0:

								if includeObservation:

									atmTensor = m.getData(l, i)
									includeObservation = self.checkMS(atmTensor, naMesan)
									dataTensor = np.dstack((dataTensor, atmTensor))
									meanArr[dataTensor.shape[2] - 1] += np.mean(atmTensor)
									stdArr[dataTensor.shape[2] - 1] += np.std(atmTensor)

							else:
								
								if includeObservation:

									atmTensor = m.getData(l, i)
									includeObservation = self.checkMS(atmTensor, naMesan)
									dataTensor = np.dstack((dataTensor, atmTensor))
									meanArr[dataTensor.shape[2] - 1] = meanArr[dataTensor.shape[2] - 2]
									stdArr[dataTensor.shape[2] - 1] = stdArr[dataTensor.shape[2] - 2]           

					if includeObservation:
						# Upload to file

						if fixdim:
							dataFile["inputData"][i, ...] = dataTensor[:,34:588][None] # img corp
						else:
							dataFile["inputData"][i, ...] = dataTensor[None]

						dataFile["outputData"][i, ...] = atmRadarVector[None]

						# Timestamp of the output radar
						dataFile["outputDataTimestamp"][i, ...] = forcastTimestamp[None]

						count += 1

						sys.stdout.write('\rDate: {} Complete upload:{}% Total sec: {}'.format(self.theDate, round((i + 1)/float(self.nIndex) * 100, 2), round(time.time() - timer), 2))
						sys.stdout.flush()

			except:
			   pass

		meanArr = meanArr/count
		stdArr = stdArr/count
		
		dataFile["mean"][...] = meanArr[None]
		dataFile["std"][...] = stdArr[None]
		dataFile["nObservations"][...] = np.array([count], dtype=np.float32)[None]

		sys.stdout.write(' NumberObsLoaded: {}/{}'.format(count, self.nIndex))
		sys.stdout.flush()

		# Close the datafile
		dataFile.close()

	def close(self):
		
		for f in self.EC_List:
			f.close()

		for f in self.MS_List:
			f.close()

		for f in self.RA_List:
			f.close()

		print '\nHDF5 datasets closed'


def main():

	###################################################################
	###################################################################
	###################################################################

	ECVarName = ['cape', 'tp', 'u700', 'u850', 'v700', 'v850']
	MSVarName = ['hcc', 'lcc', 'mcc', 'prec1h', 't', 'tcc', 'tiw']

	# Lag dic, 0 means one lag
	varLag = {'EC': 0, 'MS': 0, 'RA': 1}

	# Forcast horison
	forcastTime = 180

	# Path to save dataset
	savePathDataset = ''

	###################################################################
	###################################################################
	###################################################################

	# Number of obs to load i.e. subset. If no subset then have None
	loadNobs = None

	# Loop dataset
	dateList = []

	dateList.append(["201001", "201002", "201003", "201004", "201005", "201006", "201007", "201008", "201009", "201010", "201012"])
	dateList.append(["201101", "201102", "201103", "201104", "201105", "201106", "201107", "201108", "201109", "201110", "201111", "201112"])
	dateList.append(["201201", "201202", "201203", "201204", "201205", "201206", "201207", "201208", "201209", "201210", "201211", "201212"])
	dateList.append(["201301", "201302", "201303", "201304", "201305", "201306", "201307", "201308", "201309", "201310", "201311", "201312"])
	dateList.append(["201401", "201402", "201403", "201404", "201405", "201406", "201407", "201408", "201409", "201410", "201411", "201412"])

	dateListAll = []

	for p in dateList:

		for f in p:
			dateListAll.append(f)	

	for date in dateListAll:

		# Create dataset object and load data
		allData = periodData(ECVarName, MSVarName, date, varLag, forcastTime, loadNobs)
		allData.loadData()
		allData.createDataIndex()
		allData.loadHDF5(savePathDataset)
		allData.close()

if __name__ == '__main__':
	main()