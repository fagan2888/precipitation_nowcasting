import numpy as np
import time
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from modelResnet import resnet20
from dataFetch import handleData, meanFetch, validationFetch

def save_checkpoint(state, iter, p):
	torch.save(state, p + 'checkpoint_' + str(iter) + '.pth.tar')

def hammingLoss(yhat, y):

	predRound = np.where(yhat < 0.5, 0.0, 1.0)

	return round(1 - np.mean(np.equal(y,predRound)), 4)

def hkDiscriminant(yhat, y):

	predRound = np.where(yhat < 0.5, 0.0, 1.0)
	
	A = y[(predRound == 1)]
	A = np.count_nonzero(A == 1)
	B = y[(predRound == 1)]
	B = np.count_nonzero(B == 0)
	C = y[(predRound == 0)]
	C = np.count_nonzero(C == 1)
	D = y[(predRound == 0)]
	D = np.count_nonzero(D == 0)

	return round((A * D - B * C)/((A + C + 1.0) * (B + D + 1.0)), 4)

class modeling(object):

	def __init__(self, lR, weightdecay, datapath, valpath, bs, cuda = False, continueTrain = False, checkpointPath = None):

		self.dataset = None
		self.gpu = cuda
		self.learnR = lR
		self.objective = nn.BCEWithLogitsLoss()

		self.pathData = datapath
		self.pathDataVal = valpath
		self.batchSize = bs
		self.datasetMean = meanFetch(datapath)

		self.resnet = resnet20(pretrain = False)
		self.optimizer = torch.optim.Adam(self.resnet.parameters(), lr=self.learnR, weight_decay = weightdecay)

		self.counter = 0

		if continueTrain:
			# Here we will load the optim, model etc
			checkpoint = torch.load(checkpointPath)
			self.resnet.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(torch.load(state_dict['optimizer']))
			self.counter = checkpoint['batchCount']


		if self.gpu:
			self.resnet.cuda()
			self.objective.cuda()
			torch.backends.cudnn.benchmark = True

	def loadData(self, f):

		if not self.dataset == None:
			self.dataset.closeData()

		self.dataset = handleData(self.batchSize, self.pathData + f, self.datasetMean)
		self.nBatch = self.dataset.getNBatch()

	def loadValidationData(self):

		# Load pytensor only to memory
		self.val_x, self.val_y = validationFetch(self.pathDataVal, self.datasetMean)
		self.val_x = np.split(self.val_x, 10)

		self.valY_torch = self.makeVar(self.val_y, vol = True)


	def makeVar(self, npArr, floatVar = True, vol = False):

		torchTensor = torch.from_numpy(npArr)
		
		if floatVar:
			var = Variable(torchTensor, volatile = vol).float()
		else:
			var = Variable(torchTensor, volatile = vol).long()

		if self.gpu:
			var = var.cuda()

		return var

	def trainBatch(self, batchY, batchX):

		self.resnet.train()

		# Zero gradients
		self.optimizer.zero_grad()

		# Convert to pytensor and correct type
		imgData = self.makeVar(batchX)
		imgLabel = self.makeVar(batchY)

		pred = self.resnet(imgData)

		loss = self.objective(pred, imgLabel)

		# Calculate gradients
		loss.backward()

		#Update the pars
		self.optimizer.step()

		torch.cuda.empty_cache()

		return loss.data[0]

	def predict(self):

		self.resnet.eval()

		# Iter predictions and append
		for i, d in enumerate(self.val_x):

			atmVar = self.makeVar(d, vol = True)

			if i == 0:
				catPred = nn.Sigmoid()(self.resnet(atmVar))
			else:
				yhat = nn.Sigmoid()(self.resnet(atmVar))
				catPred = torch.cat((catPred, yhat), 0)
		
		if self.gpu:
			# If GPU we most convert to cpu because numpy no gpu support
			catPred = catPred.cpu()

		return catPred.data.numpy()

	def validate(self):

		pred = self.predict()

		validateLoss = nn.BCEWithLogitsLoss()(self.makeVar(pred, vol = True), self.valY_torch)

		return hammingLoss(pred, self.val_y), hkDiscriminant(pred, self.val_y), validateLoss.data[0]

	def getNumberBatches(self):
		return self.nBatch

def main():

	# Paths
	trainDataPath 	= ''
	valDataPath 	= ''
	pathModelSave 	= ''

	learnRate 		= 0.001
	weightDecay 	= 0.001
	epoch 			= 1000
	batchsize 		= 100
	dataFiles 		= os.listdir(trainDataPath)

	# Define model
	model = modeling(learnRate, weightDecay, trainDataPath, valDataPath, batchsize, cuda = True)
	count = model.counter
	# Init values
	avgLoss = 0
	avgLossVal = 0
	bestVal = 1.0

	# Time training
	timer = time.time()

	fileRecord = open('modelRun.txt', 'w')
	fileRecord.write('Loss,NumberBatches,Epoch,TimeData,TimeWeightUpdate' + '\n')
	fileRecord.close()

	print 'Start train!!'

	for j in range(epoch):

		for f in dataFiles:

			# Load one month
			model.loadData(f)

			# Loop over the n batches in this month
			for b in range(model.getNumberBatches()):

				timerData = time.time()
				x, y = model.dataset.getBatch(b)
				avgTimeData = round(time.time() - timerData, 2)

				timerBatch = time.time()
				lossBatch = model.trainBatch(y, x)
				avgTimeBatch = round(time.time() - timerBatch, 2)

				avgLoss = avgLoss * 0.9 + 0.1 * lossBatch

				count += 1

				summary = 'Loss: {} \t NumberBatches: {} Epoch: {} \t TimeData: {} TimeBatchTorch {}'.format(avgLoss, count, j, avgTimeData, avgTimeBatch)
				print summary + '\n'

				fileRecord = open('modelRun.txt', 'a')
				recordThis = '{},{},{},{},{}'.format(avgLoss, count, j, avgTimeData, avgTimeBatch) + '\n'
				fileRecord.write(recordThis)
				fileRecord.close()


				if count%30 == 0:

					# Save weights
					save_checkpoint({
					'batchCount': count,
					'epoch': j,
					'state_dict': model.resnet.state_dict(),
					'optimizer' : model.optimizer.state_dict(),
					}, count, pathModelSave)		

		print 'EpochAvgTime: {}'.format(round(time.time() - timer, 2)/(j + 1))
		print 'TotalTime: {}\n'.format(round(time.time() - timer, 2))
	
if __name__ == '__main__':
	main()