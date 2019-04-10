import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

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

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu2 = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(2, stride=2)
		self.avgpool1 = nn.AvgPool2d(2, stride=2)
		self.fcNew = nn.Linear(2048 * block.expansion, 2500)

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

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu2(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = self.avgpool1(x)
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