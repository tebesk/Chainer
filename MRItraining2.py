#!/usr/bin/env python
# -*- coding: utf-8 -*-
#画像一枚から512本の回答を作成

#http://kivantium.hateblo.jp/entry/2016/02/04/213050
from __future__ import print_function
import cv2
import argparse
import os

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import random
import numpy as np
from chainer import cuda,Variable,optimizers
import time
import cupy


from PIL import Image
from matplotlib import pylab as plt

# 引数の処理
parser = argparse.ArgumentParser(
    description='train convolution filters')
parser.add_argument('org', help='Path to original image')

# クラスの定義
class Conv(chainer.Chain):
	def __init__(self):
		super(Conv, self).__init__(
			# 入力・出力1ch, ksize=3
			conv1=F.Convolution2D(1, 16, 3, pad=1),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv2=F.Convolution2D(16, 1, 3, pad=1),
			conv2_1=F.Convolution2D(80, 1, 3, pad=1),
			conv3=F.Convolution2D(32, 64, 3, pad=1),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv4=F.Convolution2D(1, 32, 3, pad=1),
			conv5=F.Convolution2D(32, 64, 5, pad=2),
			conv6=F.Convolution2D(64, 32, 5, pad=2),

			l1=     F.Linear(150*128*16, 512),
			l2=     F.Linear(512, 512),
			l3=     F.Linear(128*150, 512),
			l4=     F.Linear(1360, 512),
			l5=     F.Linear(320, 512),
			norm1=	L.BatchNormalization(1),
			norm2=	L.BatchNormalization(16),
			norm3=	L.BatchNormalization(32),
			norm4=	L.BatchNormalization(64),
			norm5=	L.BatchNormalization(1360),
			norm6=	L.BatchNormalization(320),
		)

	def clear(self):
		self.loss = None
		self.accuracy = None

	def forward(self, x,layer, train=True):
		self.clear()
		if(layer ==0) :
			h = F.max_pooling_2d(F.relu(model.conv1(x)),4)
			h = self.norm2(h,test= not train)                               
			h = F.relu(model.l1(h))
			h = F.relu(model.l2(h))
		elif layer ==1 :
			h = F.spatial_pyramid_pooling_2d(F.relu(model.conv1(x)),4,F.MaxPooling2D)
			h = self.norm5(h,test= not train)
			h = F.relu(model.l4(h))
			h = F.relu(model.l2(h))
		elif layer ==2 :
			h = F.average_pooling_2d(F.relu(model.conv1(x)),4)
			h = self.norm2(h,test= not train)                               
			h = F.relu(model.l1(h))
			h = F.relu(model.l2(h))
		elif layer ==3 :
			h = F.max_pooling_2d(F.relu(model.conv1(x)),2)
			h = self.norm2(h,test= not train)
			h = F.max_pooling_2d(F.relu(model.conv2(h)),2)
			h = self.norm1(h,test= not train)
			h = F.relu(model.l3(h))
			h = F.relu(model.l2(h))
		elif layer ==4 :
			h = F.max_pooling_2d(F.relu(model.conv4(x)),2)
			h = self.norm3(h,test= not train)
			h = F.spatial_pyramid_pooling_2d(F.relu(model.conv3(h)),2,F.MaxPooling2D)
			h = self.norm6(h,test= not train)
			h = F.relu(model.l5(h))
			h = F.relu(model.l2(h))
		elif layer ==5 :
			h = F.average_pooling_2d(F.relu(model.conv1(x)),2)
			h = self.norm2(h,test= not train)
			h = F.average_pooling_2d(F.relu(model.conv2(h)),2)
			h = self.norm1(h,test= not train)
			h = F.relu(model.l3(h))
			h = F.relu(model.l2(h))
		return h

	def calc_loss(self, x, t, layer,train=True):
		self.clear()
		if(layer ==0) :
			h = F.max_pooling_2d(F.relu(model.conv1(x)),4)
			h = self.norm2(h,test= not train)                               
			h = F.relu(model.l1(h))
			h = F.relu(model.l2(h))
		elif layer ==1 :
			h = F.spatial_pyramid_pooling_2d(F.relu(model.conv1(x)),4,F.MaxPooling2D)
			h = self.norm5(h,test= not train)
			h = F.relu(model.l4(h))
			h = F.relu(model.l2(h))
		elif layer ==2 :
			h = F.average_pooling_2d(F.relu(model.conv1(x)),4)
			h = self.norm2(h,test= not train)                               
			h = F.relu(model.l1(h))
			h = F.relu(model.l2(h))
		elif layer ==3 :
			h = F.max_pooling_2d(F.relu(model.conv1(x)),2)
			h = self.norm2(h,test= not train)
			h = F.max_pooling_2d(F.relu(model.conv2(h)),2)
			h = self.norm1(h,test= not train)
			h = F.relu(model.l3(h))
			h = F.relu(model.l2(h))
		elif layer ==4 :
			h = F.max_pooling_2d(F.relu(model.conv4(x)),2)
			h = self.norm3(h,test= not train)
			h = F.spatial_pyramid_pooling_2d(F.relu(model.conv3(h)),2,F.MaxPooling2D)
			h = self.norm6(h,test= not train)
			h = F.relu(model.l5(h))
			h = F.relu(model.l2(h))
		elif layer ==5 :
			h = F.average_pooling_2d(F.relu(model.conv1(x)),2)
			h = self.norm2(h,test= not train)
			h = F.average_pooling_2d(F.relu(model.conv2(h)),2)
			h = self.norm1(h,test= not train)
			h = F.relu(model.l3(h))
			h = F.relu(model.l2(h))
			
		'''
		print (h.data.shape)
		
		#t=t.T.data
		print (t.data)
		
		print (h.data.shape)
		#print (h.data.T)
		print (t.data.shape)

		'''
		loss = F.mean_squared_error(h, t)
		return loss


for layer in range(6):

	#Root file
	Ans_PATH= "MRI/test" 
	Training_PATH= "MRI/re_move_ivus"
	Result_PATH= "1601025_"+str(layer)+"/"
	
	if os.path.isdir(Result_PATH)==False:
		os.mkdir(Result_PATH) 
	
	### Read answer image
	Ansfiles = os.listdir(Training_PATH)

	# 学習対象のモデル作成
	model = Conv()
	chainer.cuda.get_device(0).use()  # Make a specified GPU current
	model.to_gpu()  # Copy the model to the GPU
	
	# 最適化の設定
	optimizer = optimizers.Adam()
	optimizer.setup(model)

	start = time.time()
	

	
	#学習の開始
	for seq in range(4000):
		filenames= random.sample(Ansfiles,50)
		for filename in filenames:
			#opencv file read
			#t_img = np.array( Image.open(Training_PATH+"/"+filename) )
			
			#画像ごとにTraining実施			
			train_image = chainer.Variable(cuda.cupy.asarray([[cv2.imread(Training_PATH+"/"+filename, 0)/255.0]], dtype=np.float32))
			target = chainer.Variable(cuda.cupy.asarray((cv2.imread(Ans_PATH+"/"+filename, 0)/255.0).reshape(1,512), dtype=np.float32))
								
			loss = model.calc_loss(train_image, target, layer)
			model.zerograds()
			loss.backward()
			optimizer.update()
			

		if seq%20==0:
			elapsed_time = time.time() - start
			print("{}: {}".format(seq, loss.data))
			temp= seq/200
			temp2= temp*200
			
			if os.path.isdir(Result_PATH+str(temp2))==False:
				os.mkdir(Result_PATH+str(temp2)) 
			f = open(Result_PATH+str(temp2)+"/a.txt","a")
			f.write("{}: {}".format(seq, loss.data))
			f.write("\nelapsed_time:{0}sec\n".format(elapsed_time))
			f.close()

		if seq%200==0:
			if os.path.isdir(Result_PATH+str(seq))==False:
				os.mkdir(Result_PATH+str(seq))
				
			for filename in Ansfiles:
				#forsave_img = cv2.imread(Training_PATH+"/"+filename, 0)
				train_image = chainer.Variable(cuda.cupy.asarray([[cv2.imread(Training_PATH +"/"+filename, 0)/255.0]], dtype=np.float32))
				trained = model.forward(train_image,layer).data[0]*255
				#img4 = cuda.to_cpu(trained)
				#forsave_img = cv2.hconcat([img4, forsave_img]) 
				#forsave_img = cv2.hconcat([img4, forsave_img]) 
				#forsave_img = cv2.hconcat([img4, forsave_img]) 
				cv2.imwrite(Result_PATH+str(seq)+"/"+filename, cuda.to_cpu(trained))
			chainer.serializers.save_hdf5(Result_PATH+str(seq)+"/161025.model", model)
			
	# 学習finish
	if os.path.isdir(Result_PATH+str(seq))==False:
		os.mkdir(Result_PATH+str(seq))    
		
	for filename in Ansfiles:
		forsave_img = cv2.imread(Training_PATH+"/"+filename, 0)
		train_image = chainer.Variable(cuda.cupy.asarray([[cv2.imread(Training_PATH +"/"+filename, 0)/255.0]], dtype=np.float32))
		trained = model.forward(train_image,layer).data[0]*255
		#img4 = cuda.to_cpu(trained)
		#forsave_img = cv2.hconcat([img4, forsave_img]) 
		#forsave_img = cv2.hconcat([img4, forsave_img]) 
		#forsave_img = cv2.hconcat([img4, forsave_img]) 
		cv2.imwrite(Result_PATH+str(seq)+"/"+filename, cuda.to_cpu(trained))
	
	
#chainer.serializers.save_hdf5(Result_PATH+'161025.model', model)
