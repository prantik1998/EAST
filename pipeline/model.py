import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import sys


import math
	
class EAST(nn.Module):
	def __init__(self,model_type="PVA",pretrained=True):
		super(EAST, self).__init__()
		print(model_type)
		if model_type=="PVA":
			from .Models.PVA import PVA ,merge,output
			self.extractor = PVA()
			self.merge     = merge()
			self.output    = output()
		elif model_type=="ResNet":
			from .Models.ResNet import Net ,merge,output
			self.extractor = Net()
			self.merge     = merge()
			self.output    = output()

		elif model_type=="DarkNet":
			from .Models.Darknet import Net ,merge,output
			self.extractor = Net()
			self.merge     = merge()
			self.output    = output()
		else :
			from .Models.vgg import VGG ,merge,output
			self.extractor = VGG()
			self.merge     = merge()
			self.output    = output()
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))
		

