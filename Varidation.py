# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import math
import time

ans_thresh =10

img_path = "NN_calc_result/vsMRI"
ans_path = "MRI/test"
folders=os.listdir(img_path)
for folder in folders:
	f_precision = open(img_path+"/"+folder+'_precision.csv','a')
	f_recall = open(img_path+"/"+folder+'_recall.csv','a')
	f_accuracy = open(img_path+"/"+folder+'_accuracy.csv','a')
	print folder
	temp_folder=os.listdir(img_path+"/"+folder)
	for detail_folder in temp_folder :			
			
		for thresh in range(10,100):
			TP = 0
			TN = 0
			FP = 0
			FN = 0
				
			temp_detail_folder=os.listdir(img_path+"/"+folder+"/"+detail_folder)
			for image_name in temp_detail_folder :
				root, ext = os.path.splitext(img_path+"/"+folder+"/"+detail_folder+"/"+image_name)
				#print img_path+"/"+folder+"/"+detail_folder+"/"+image_name
				if ext==".bmp":
					learned_img_path = img_path+"/"+folder+"/"+detail_folder+"/"+image_name
					ans_img = cv2.imread(ans_path+"/"+image_name)
					img = cv2.imread(learned_img_path)
					
					for i in range(512):
						test = 0
						ans  = 0
						
						pixelValue = img[i, 0]
						ans_pixelValue = ans_img[i,0]
							
						if pixelValue[0]> thresh:
							test = 1
						if ans_pixelValue[0]> ans_thresh:
							ans  = 1
							
						if test == ans and test == 1 :
							TP = TP + 1
						elif test == ans and test == 0 :
							TN = TN + 1
						elif test != ans and test == 1 :
							FP = FP + 1
						elif test != ans and test == 0 :
							FN = FN + 1
						else :
							print "error at "+learned_img_path
				#print "TP:"+str(TP)+"  TN:"+str(TN)+"  FP:"+str(FP)+"  FN:"+str(FN)
			if TP+TN+FP+FN==0:
				Accuracy = -1
			else:
				Accuracy = 1.0*(TP+TN)/(1.0*(TP+TN+FP+FN))
				
				
			if TP+FP==0:
				Precision = -1
			else:
				Precision = 1.0*TP/ (1.0*(TP+FP))
				
				
			if TP+FN==0:
				Recall = -1
			else:
				Recall = 1.0*TP/ (1.0*(TP+FN))
				
			f_precision.write("%s,"%Precision)
			f_recall.write("%s,"%Recall)
			f_accuracy.write("%s,"%Accuracy)
	
		f_precision.write("\n")
		f_recall.write("\n")
		f_accuracy.write("\n")
	f_precision.close()
	f_recall.close()
	f_accuracy.close()
	
					
			
		
		
	
	
	
	
	
	

