import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
	def __init__(self,inc):
		super(Block,self).__init__()
		self.layer1=nn.Sequential(nn.Conv2d(inc,inc//2,1),nn.BatchNorm2d(inc//2),nn.LeakyReLU(0.1))
		self.layer2=nn.Sequential(nn.Conv2d(inc//2,inc,3,padding=1),nn.BatchNorm2d(inc),nn.LeakyReLU(0.1))
	def forward(self,x):
		return x + self.layer2(self.layer1(x))


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.LeakyReLU(0.1))
		self.conv2=nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.1))
		self.res1=Block(64)
		self.conv3=nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU(0.1))
		self.res2=nn.Sequential(Block(128))
		self.conv4=nn.Sequential(nn.Conv2d(128,256,3,stride=2,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.1))
		self.res3=nn.Sequential(Block(256))
		self.conv5=nn.Sequential(nn.Conv2d(256,512,3,stride=2,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1))
		self.res4=nn.Sequential(Block(512))
		self.conv6=nn.Sequential(nn.Conv2d(512,512,3,stride=2,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1))
		self.res5=nn.Sequential(Block(512))
				

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m,nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 1)
	def forward(self,x):
		out=[]
		x=self.res1((self.conv2(self.conv1(x))))
		out.append(x)

		x=self.res2(self.conv3(x))
		out.append(x)

		x=self.res3(self.conv4(x))
		out.append(x)

		x=self.res4(self.conv5(x))
		out.append(x)

		x=self.res5(self.conv6(x))
		out.append(x)
		return out[1:]




class merge(nn.Module):
	def __init__(self):
		super(merge,self).__init__()
		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self,x):
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[2]), 1)
		y = self.relu1(self.bn1(self.conv1(y)))		
		y = self.relu2(self.bn2(self.conv2(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		
		y = self.relu7(self.bn7(self.conv7(y)))
		return y

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		geo   = torch.cat((loc, angle), 1) 
		return score, geo


		

