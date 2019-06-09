import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .train import train
from .eval import eval_model
from .detect import detect,plot_boxes
from .model import EAST


class pipeline_manager():

	def __init__(self,config):
		self.config=config
		self.cuda=torch.cuda.is_available()

	def train(self,model_path,model_name='PVA'):

		train(self.config['train_img_path'], self.config['train_gt_path'], self.config['pths_path'], self.config['batch_size'], self.config['lr'], self.config['num_workers'], self.config['epoch_iter'], self.config['save_interval'],self.cuda,model_path,model_name)

	def test(self,model_path=None,model_name='PVA'):
		if model_path==None:
			model_path=self.config['model_name']
		eval_model(model_path,self.config['test_img_path'],self.config['submit_path'],self.cuda,model_name)

	def detect(self,model_path,img_path,res_path):
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model = EAST().to(device)
		model.load_state_dict(torch.load(model_path))
		model.eval()
		img = Image.open(img_path)
		
		boxes = detect(img, model, device)
		plot_img = plot_boxes(img, boxes)	
		plot_img.save(res_path)
