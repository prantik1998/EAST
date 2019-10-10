import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os

from .train import train
from .eval import eval_model,eval_model_dir
from .detect import detect,plot_boxes
from .model import EAST


class pipeline_manager():

	def __init__(self,config):
		self.config=config
		self.cuda=torch.cuda.is_available()

	def train(self,model_name):
		if 'model_path' not in self.config.keys():
			self.config['model_path']=False
		if model_name is not None:
			self.config["model_name"] = model_name
		train(self.config,self.cuda)

	def test(self,model_name):
		if model_name is not None:
			self.config["model_name"] = model_name
		eval_model(self.config,self.cuda)

	def test_dir(self,model_name):
		if model_name is not None:
			self.config["model_name"] = model_name
		eval_model_dir(self.config,self.cuda)

	def detect(self,model_path,img_path,res_path):
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model = EAST().to(device)
		model.load_state_dict(torch.load(model_path))
		model.eval()
		img = Image.open(img_path)
		
		boxes = detect(img, model, device)
		plot_img = plot_boxes(img, boxes)	
		plot_img.save(res_path)

	
		