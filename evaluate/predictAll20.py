import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py as h5
import math
import os


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.dropout = nn.Dropout(inplace=True)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		# Dropout as in Widepaper
		out = self.dropout(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNet20(nn.Module):

	def __init__(self, block, layers):
		self.inplanes = 64
		super(ResNet20, self).__init__()

		self.convNew1 = nn.Conv2d(15, 8, kernel_size = 2, stride = 2, padding = 1)
		self.relu1= nn.ReLU(inplace=True)
		self.convNew2 = nn.Conv2d(8, 3, kernel_size = 3, stride = 1, padding = 1)
		self.relu2= nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(9)
		self.fcNew = nn.Linear(512 * block.expansion, 2500)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.convNew1(x)
		x = self.relu1(x)
		x = self.convNew2(x)
		x = self.relu2(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu3(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fcNew(x)

		return x

def resnet20(pretrain = False):

	model20 = ResNet20(BasicBlock, [2, 2, 2, 2])

	if pretrain:

		# Load the pretrained for resnet 18 layers
		model18params = torch.load('resnet18.pkl')
		model20_dict = model20.state_dict()

		# Replace the random init weights with pretrained
		for name, par in model18params.items():
			if name not in model20_dict:
				# Jump to next loop
				continue
			else:
				model20_dict[name].copy_(par)

		model20.load_state_dict(model20_dict)

	return model20

def meanFetch(datapathMean):

	files = os.listdir(datapathMean)

	for f in files:

		atmPath = datapathMean + f
		atmData = h5.File(atmPath, 'r')

		if files.index(f) == 0:
			Trainmean = atmData['mean'][0]
		else:
			Trainmean += atmData['mean'][0]


		atmData.close()

	return Trainmean/len(files)

def makePyVar(x):

	torchTensor = torch.from_numpy(x)
	var = Variable(torchTensor, volatile = True).float()

	return var.cuda()

def makePrediction(modelPred, x):

	pred = modelPred(x)
	pred = nn.Sigmoid()(pred)
	pred = pred.cpu()

	return pred.data.numpy()
	
def makeHD(datasetPath, savePath, dataname):

	dataFiles = os.listdir(datasetPath)

	nObs = 0

	for d in dataFiles:

		atmData = h5.File(datasetPath + d, 'r')

		nObs += atmData['nObservations'][0][0]

	dataSet = h5.File(savePath + dataname, mode='w')

	predShape = (nObs, 2500)
	trueShape = (nObs, 2500)
	timeShape = (nObs, 1)
	nObsUploaded = (1, 1)

	dataSet.create_dataset("prediction", predShape, np.float32)
	dataSet.create_dataset("trueLabel", trueShape, np.float32)
	dataSet.create_dataset("timestamp", timeShape, np.float32)
	dataSet.create_dataset("n_Observations", nObsUploaded, np.float32)

	return dataSet, nObs

def main():

	filename = ''

	# Model save path
	checkpointPath = ''

	# Data to make forcast
	datasetPath = ''
	
	dataFiles = os.listdir(datasetPath)

	# Create HD file to store the predictions
	savePath = ''
	saveData, totalObs = makeHD(datasetPath, savePath, filename)

	checkpoint = torch.load(checkpointPath)	
	model = resnet20()
	model.load_state_dict(checkpoint['state_dict'])
	model.cuda()
	model.eval()

	count = 0

	print "Start prediction!"
	for d in dataFiles:

		atmData = h5.File(datasetPath + d, 'r')

		for i in range(atmData['nObservations'][0][0]):

			x = atmData['inputData'][i, ...]
			x = x.T
			x = makePyVar(x[None])

			y = atmData['outputData'][i, ...]
			time = atmData['outputDataTimestamp'][i, ...]
			pred = makePrediction(model, x)

			saveData["prediction"][count, ...] = pred
			saveData["trueLabel"][count, ...] = y[None]
			saveData["timestamp"][count, ...] = time[None]

			count += 1

			print "Iter {}/{}".format(count, totalObs)

		atmData.close()

	saveData.close()

if __name__ == '__main__':
	main()