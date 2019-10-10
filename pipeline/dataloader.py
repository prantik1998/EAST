import os
import sys

from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils import data

def rotate_mat(theta):
	return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def rotate_img(img,vertices,angle_range = 10):
	anchor = np.array([img.width-1,img.height-1])
	angle = angle_range*(np.random.rand()*2-1)
	img = img.rotate(angle_range,Image.BILINEAR)
	for i,v in enumerate(vertices):
		vertices[i,:] = get_rotate_vertices(v,-angle / 180 * np.pi,anchor)
	return img,vertices


def adjust_height(img,vertices,ratio = 0.2):
	ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
	img = img.resize((img.width,int(ratio_h*img.height)), Image.BILINEAR)
	vertices[:,1] = vertices[:,1] * ratio_h
	return img,vertices.astype(np.int32)


def rotate_all_vertices(rot_mat,p,length):
	x,y = np.meshgrid(np.arange(length),np.arange(length))
	x_lin,y_lin = x.reshape((1,x.size)),y.reshape((1,y.size))
	coord_mat = np.concatenate((x_lin, y_lin), 0)
	coord_mat[0] = coord_mat[0] - p[0]
	coord_mat[1] = coord_mat[1] - p[1]
	rot_coord = np.dot(rot_mat,coord_mat)
	rot_coord[0] += p[0]
	rot_coord[1] += p[1]
	rot_x = rot_coord[0,:].reshape(x.shape)
	rot_y = rot_coord[1,:].reshape(y.shape)
	return rot_x,rot_y

def get_rotate_vertices(v,theta,anchor = None):
	rot_mat = rotate_mat(theta)
	if anchor is None:
		return np.dot(v-v[:1,:],rot_mat)+v[:1,:]
	else :
		return np.dot(v-anchor,rot_mat)+anchor		

def cal_error(v):
	vertices = v.copy()
	x_max,x_min,y_max,y_min = get_boundary(vertices)
	p0,p1,p2,p3 = vertices
	return distance(p0,np.array([x_min,y_min]))+distance(p1,np.array([x_max,y_min]))+distance(p2,np.array([x_max,y_max]))+distance(p3,np.array([x_min,y_max]))


def get_boundary(v):
	x_max = max(v[:,0])
	x_min = min(v[:,0])
	y_max = max(v[:,0])
	y_min = min(v[:,0])
	return x_max,x_min,y_max,y_min


def find_min_rect_angle(v):
	x_max,x_min,y_max,y_min = get_boundary(v)
	area = []
	min_error = float("inf")
	best_angle = 180
	for i in range(-90,90,1):
		rot_vert = get_rotate_vertices(v,i*(np.pi/180))
		temp_error = cal_error(rot_vert)
		if temp_error<min_error:
			min_error = temp_error
			best_angle = i*(np.pi/180)
	return best_angle

def distance(p1,p2):
	p2,p1 = np.array(p2),np.array(p1)
	return np.sqrt(np.sum((p2-p1)**2))

def move_points(v,ind,offset,ref_len):
	print(v)
	a = input()
	ind = (np.array(ind)+offset)%4
	r1,r2 = ref_len[ind[0]],ref_len[ind[1]]
	p1,p2 = v[ind[0]],v[ind[1]]
	length = distance(p1,p2).astype(np.float32)
	print(length)
	if length>1:
		len_x = p2[0]-p1[0]
		len_y = p2[1]-p1[1]
		v[ind[0]] -= (r1/length)*len_x
		v[ind[1]] -= (r2/length)*len_y
	return v.astype(np.int32)

def shrink_poly(v,shrink_factor = 0.3):
	#shrink_factor as suggested by the paper
	
	ref_len = shrink_factor*np.array([min(distance(v[i],v[(i+1)%4]),distance(v[i],v[(i+2)%4])) for i in range(4)]) 
	if distance(v[0],v[1]) + distance(v[2],v[3]) > distance(v[2],v[1]) + distance(v[0],v[3]):
		offset = 0
	else :
		offset = 1
	vert = v.copy()
	vert = move_points(vert,[0,1],offset,ref_len)
	vert = move_points(vert,[2,3],offset,ref_len)
	vert = move_points(vert,[1,2],offset,ref_len)
	vert = move_points(vert,[0,4],offset,ref_len)
	return vert

def extract_vertices(pth):

	f = open(pth,"r")
	data = f.read()
	data = [i.lstrip("ï»¿") for i in data.split('\n') if i !='']
	data = [i.split(',') for i in data]
	data = np.array(data)
	vertices = np.array(np.array([list(map(int,i[:8])) for i in data])).astype(np.int32)
	labels = np.array([0 if '###' in i  else 1 for i in data ])
	return vertices.reshape(labels.shape[0],4,2),labels

def is_cross_text(start_x,start_y, length, vertices,labels):
	
	if vertices.size == 0:
		return False
	a = np.array([start_x, start_y, start_x + length, start_y,start_x + length, start_y + length, start_x, start_y + length]).reshape((4,2))
	p1 = Polygon(a).convex_hull
	for i,vertice in enumerate(vertices):
		if labels[i] == 1:
			p2 = Polygon(vertice).convex_hull
			inter = p1.intersection(p2).area
			if 0.01 <= inter / p2.area <= 0.99: 
				return True
	return False
		

def crop_img(img, vertices, labels, length):
	h, w = img.height, img.width
	if min(h,w)<length:
		if min(h,w) == h:
			img.resize((int(w*length/h),length),Image.BILINEAR)
		else:
			img.resize((length,int(h*length/w)),Image.BILINEAR)
	ratio_h,ratio_w = img.height/h,img.width/w
	new_vertices = np.zeros(vertices.shape)
	print(vertices[:,:,0])
	new_vertices[:,:,0] = vertices[:,:,0]*ratio_w
	new_vertices[:,:,1] = vertices[:,:,1]*ratio_h
	remain_w,remain_h = img.width-length,img.height-length
	start_x = int(remain_w*np.random.rand())
	start_y = int(remain_h*np.random.rand())
	time = 0
	cross_text = is_cross_text(start_x,start_y,length,vertices,labels)
	while cross_text and time<1000:
		time+=1
		start_x = int(remain_w*np.random.rand())
		start_y = int(remain_h*np.random.rand())
		cross_text = is_cross_text(start_x,start_y,length,vertices,labels)
	region = img.crop([start_x,start_y,start_x+length,start_y+length])
	new_vertices[:,:,0] -= start_x
	new_vertices[:,:,1] -= start_y
	v = []
	lbl = []
	for i,vert in enumerate(new_vertices):
		if np.max(vert)<=length:
			v.append(vert)
			lbl.append(labels[i])
	v = np.array(v)	
	v[v<0] = 0
	return region, v,lbl


def resize_img(pth,vertices,length):
	img = Image.open(pth)
	img = cv2.imread(pth)
	h,w = img.shape[:2]
	max_side = max(h,w)
	dim = (int((w * length)/max_side), int((h * length)/max_side))
	img = cv2.resize(img,dim,cv2.INTER_AREA)
	ratio = length/max_side
	resize_img = np.ones((length,length,3),dtype = np.float32)*np.mean(img)
	start_x,start_y = (length-img.shape[0])//2,(length - img.shape[1])//2
	resize_img[start_x:start_x+img.shape[0],start_y:start_y+img.shape[1]] = img
	vertices[:,1::2] = vertices[:,1::2]*ratio+start_x
	vertices[:,::2] = vertices[:,::2]*ratio+start_y
	assert np.sum(vertices[:,1::2]>resize_img.shape[0]) == 0 and np.sum(vertices[:,::2]>resize_img.shape[1]) == 0
	return resize_img.transpose(2,0,1),vertices

def get_score_geo(img,vertices,labels,length,scale):
	dim = (int(img.width*scale),int(img.height*scale))
	score_map = np.zeros((dim[0],dim[1],1)).astype(np.float32) #representing text scores
	geo_map = np.zeros((dim[0],dim[1],5)).astype(np.float32)
	ignored_map = np.zeros((dim[0],dim[1],1)).astype(np.float32)

	index = np.arange(0, length, int(1/scale))
	index_x, index_y = np.meshgrid(index, index)
	polys = []
	ignored_poly = []

	for i,v in enumerate(vertices):
		if labels[i] == 0:
			ignored_poly.append(np.around(scale*v).astype(np.int32))
			continue
		poly = np.around(scale*shrink_poly(v)).astype(np.int32)
		polys.append(poly)
		temp_mask = np.zeros(score_map.shape[:-1], np.float32)
		cv2.fillPoly(temp_mask, [poly], 1)

		theta = find_min_rect_angle(v)
		rot_mat = rotate_mat(theta)

		rotate_vertices = get_rotate_vertices(v,theta)
		x_max,x_min,y_max,y_min = get_boundary(rotate_vertices)
		rot_x,rot_y = rotate_all_vertices(rot_mat,v[0],length)

		d1 = rot_y-y_min
		d1[d1<0] = 0
		d2 = y_max-rot_y
		d2[d2<0] = 0
		d3 = rot_x-x_min
		d3[d3<0] = 0
		d4 = x_max-rot_x
		d4[d4<0] = 0
		geo_map[:,:,0] += d1[index_y, index_x] * temp_mask
		geo_map[:,:,1] += d2[index_y, index_x] * temp_mask
		geo_map[:,:,2] += d3[index_y, index_x] * temp_mask
		geo_map[:,:,3] += d4[index_y, index_x] * temp_mask
		geo_map[:,:,4] += theta * temp_mask


	cv2.fillPoly(ignored_map, ignored_poly, 1)
	cv2.fillPoly(score_map, polys, 1)
	return torch.Tensor(score_map).permute(2,0,1), torch.Tensor(geo_map).permute(2,0,1), torch.Tensor(ignored_map).permute(2,0,1)

class custom_dataset(data.Dataset):
	def __init__(self, img_dir, gt_dir,length=512,scale=0.25):
		super(custom_dataset, self).__init__()
		self.img_dir = img_dir
		self.gt_dir  = gt_dir
		self.img_name = sorted(os.listdir(img_dir))
		self.length = length
		self.scale = scale

	def __len__(self):
		return len(self.img_name)

	def __getitem__(self, index):
		img_name = self.img_name[index]
		gt_name = "gt_"+"".join(img_name.split(".")[:-1])+".txt"
		vertices, labels = extract_vertices((self.gt_dir+gt_name))
		img = Image.open(os.path.join(self.img_dir,img_name))
		img, vertices = adjust_height(img, vertices) 
		img, vertices = rotate_img(img, vertices)
		img,vertices = crop_img(img,vertices,labels,self.length)
		transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

		score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels,self.length,self.scale)
		return transform(img), score_map, geo_map, ignored_map

if __name__=="__main__":
	gt_img = "D:/IC15/train_gt/gt_img_645.txt"
	img_pth = "D:/IC15/train_img/img_645.jpg"
	vertices, labels = extract_vertices(gt_img)
	img = cv2.imread(img_pth)
	print(vertices.shape)
	cv2.drawContours(img,vertices,-1,(0,255,0),3)
	cv2.imshow("Image",img)
	cv2.waitKey(0)
	print(vertices)
	img = Image.open(img_pth)
	print(img.size)
	img,vertices,label = crop_img(img,vertices,labels,512)
	print(img.size)
	img.show()
	print(vertices)
	