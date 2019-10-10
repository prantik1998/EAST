import os
import time
import shutil

import numpy as np

import torch

from .model import EAST
from .detect import detect_dataset
from .evaluate.create_zip import main as create_zip

def eval_model(config,cuda,save_flag=True):

	model_path = config["model_path"]
	test_img_path = config["test_img_path"]
	submit_path = config["submit_path"]
	model_name = config["model_name"]

	print(f"model_path:- {model_path}")

	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if cuda else "cpu")
	model = EAST(model_name,False).to(device)
	if model_path:
		model.load_state_dict(torch.load(model_path))
	else:
		pass
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	# os.system('python main.py create-zip')
	# os.system('python main.py script')
	create_zip(config)

	pth = config['evaluate_path']+"/script.py"
	os.system("python "+pth)

	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)

def eval_model_dir(config,cuda,save_flag=False):
	model_dir = config["model_dir"]

	print("Model_dir")
	print(model_dir)
	print("")
	mod_pth = [os.path.join(model_dir,i) for i in sorted(os.listdir(model_dir))]
	for m in mod_pth:
		if not os.path.isdir(m):	
			config["model_path"] = m
			eval_model(config,cuda,save_flag)

if __name__ == '__main__': 
	model_name = 'D:/save_path/model_epoch_21.pth'
	test_img_path = os.path.abspath('D:/IC15/test_img')
	submit_path = './submit'
	eval_model(model_name, test_img_path, submit_path)