import numpy as np
import cv2
if __name__=="__main__":
	for i in range(1,101):
		bounding_box=open("submit/res_img_"+str(i)+".txt","r").read().split("\n")
		bounding_box=[np.array(eval(i)).reshape(4,1,2) for i in bounding_box if i!='']
		img=cv2.imread("D:/IC15/test_img/img_"+str(i)+".jpg")
		cv2.imshow('image',img)
		cv2.waitKey(0)	
		print(len(bounding_box))
		cv2.drawContours(img,bounding_box, -1, (0,255,0), 3)
		cv2.imshow('image',img)
		cv2.waitKey(0)