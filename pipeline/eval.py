import time
import torch
import subprocess
import os

from .model import EAST
from .detect import detect_dataset

import numpy as np
import shutil


def eval_model(model_path, test_img_path, submit_path, cuda,model_name,save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if cuda else "cpu")
	model = EAST(model_name,False).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python /home/mayank/Documents/EAST/pipeline/evaluate/script.py –g=/home/mayank/Documents/EAST/pipeline/evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


