#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
			conv1=F.Convolution2D(1, 32, 3, pad=1),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv2=F.Convolution2D(32, 1, 3, pad=1),
			conv3=F.Convolution2D(1, 32, 5, pad=2),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv4=F.Convolution2D(32, 1, 5, pad=2),
			l3=     F.Linear(600, 600),
			l4=     F.Linear(600, 1),
			norm1=L.BatchNormalization(1),
		)

	def clear(self):
		self.loss = None
		self.accuracy = None

	def forward(self, x,layer,train=True):
		self.clear()
		
		h = F.relu(model.conv1(x))
		h = F.relu(model.conv2(h))
		h = F.relu(model.l3(h))
		h = F.relu(model.l4(h))
		
		return h

	def calc_loss(self, x, t,layer,train=True):
		self.clear()
		h = F.relu(model.conv1(x))
		h = F.relu(model.conv2(h))
		h = F.relu(model.l3(h))
		h = F.relu(model.l4(h))
			
		loss = F.mean_squared_error(h, t)
		return loss


for layer in range(1):

	#Root file
	Ans_PATH= "MRI/kde_ans_gray" 
	Training_PATH= "MRI/re_move_ivus"
	Result_PATH= "160930_"+str(layer)+"/"
	
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
	for seq in range(2500):
		filenames= random.sample(Ansfiles,200)
		for filename in filenames:
			#opencv file read
			#t_img = np.array( Image.open(Training_PATH+"/"+filename) )
			trn_img = cv2.imread(Training_PATH+"/"+filename, 0)
			ans_img = cv2.imread(Ans_PATH+"/"+filename,0)
			
			#画像内の各ThetaごとにTraining実施
			for theta in range(512):
				#temp =trn_img[theta: theta+1, 0:599]
				train_image = chainer.Variable(cuda.cupy.asarray([[trn_img[theta: theta+1, 0:600]/255.0]], dtype=np.float32))
				target = chainer.Variable(cuda.cupy.asarray([[(ans_img[theta, 0])/255.0]], dtype=np.float32))
		
				loss = model.calc_loss(train_image, target, layer)
				model.zerograds()
				loss.backward()
				optimizer.update()
			
		print (seq)
		if seq%20==0:
			elapsed_time = time.time() - start
			print("{}: {}".format(seq, loss.data))
			temp= seq/500
			temp2= temp*500
			
			if os.path.isdir(Result_PATH+str(temp2))==False:
				os.mkdir(Result_PATH+str(temp2)) 
			f = open(Result_PATH+str(temp2)+"/a.txt","a")
			f.write("{}: {}".format(seq, loss.data))
			f.write("\nelapsed_time:{0}sec\n".format(elapsed_time))
			f.close()
		if seq%500==0:
			if os.path.isdir(Result_PATH+str(seq))==False:
				os.mkdir(Result_PATH+str(seq))    		
			for filename in Ansfiles:
				trn_img = cv2.imread(Training_PATH+"/"+filename, 0)
				for theta in range(512):
					train_image = chainer.Variable(cuda.cupy.asarray([[trn_img[theta: theta+1, 0:600]/255.0]], dtype=np.float32))
					trained = model.forward(train_image,layer).data[0][0]*255
					trn_img[theta,0]=trained
					trn_img[theta,1]=trained
					trn_img[theta,2]=trained
				cv2.imwrite(Result_PATH+str(seq)+"/"+filename, trn_img)
			chainer.serializers.save_hdf5(Result_PATH+str(seq)+"/160930.model", model)
			

	# 学習結果の表示
	print(model.conv1.W.data[0][0])

	if os.path.isdir(Result_PATH+str(seq))==False:
		os.mkdir(Result_PATH+str(seq))    
		
	for filename in Ansfiles:
		trn_img = cv2.imread(Training_PATH+"/"+filename, 0)
		for theta in range(512):
			train_image = chainer.Variable(cuda.cupy.asarray([[trn_img[theta: theta+1, 0:600]/255.0]], dtype=np.float32))
			trained = model.forward(train_image,layer).data[0][0]
			trn_img[theta,0]=trained
			trn_img[theta,1]=trained
			trn_img[theta,2]=trained
		cv2.imwrite(Result_PATH+str(seq)+"/"+filename, trn_img)
	
	
chainer.serializers.save_hdf5('160930.model', model)
