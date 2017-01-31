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

# 引数の処理
parser = argparse.ArgumentParser(
    description='train convolution filters')
parser.add_argument('org', help='Path to original image')

# クラスの定義
class Conv(chainer.Chain):
	def __init__(self):
		super(Conv, self).__init__(
			# 入力・出力1ch, ksize=3
			conv3_32=F.Convolution2D(3, 32, 5, pad=2),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv32_3=F.Convolution2D(32, 3, 5, pad=2),
			conv3=F.Convolution2D(3, 64, 5, pad=2),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv4=F.Convolution2D(64, 3, 5, pad=2),
			conv7=F.Convolution2D(3, 64, 3, pad=1),#conv1=F.Convolution2D(1, 32, 3, pad=1),
			conv8=F.Convolution2D(64, 3, 3, pad=1),
			norm_ch3=L.BatchNormalization(3),
		)

	def clear(self):
		self.loss = None
		self.accuracy = None

	def forward(self, x,layer,train=True):
		self.clear()

		#print(x.data.shape)
		h = F.relu(model.conv3_32(x))
		h = F.relu(model.conv32_3(h))
		h = F.relu(self.norm_ch3(h,test= not train))
		
		h = F.relu(model.conv3_32(h))
		h = F.relu(model.conv32_3(h))
		h = F.relu(self.norm_ch3(h,test= not train))
		
		if layer > 0:
			h = F.relu(model.conv3_32(h))
			h = F.relu(model.conv32_3(h))
			h = F.relu(self.norm_ch3(h,test= not train))
		if layer > 1:
			h = F.relu(model.conv3_32(h))
			h = F.relu(model.conv32_3(h))
			h = F.relu(self.norm_ch3(h,test= not train))
		if layer > 2:
			h = F.relu(model.conv3_32(h))
			h = F.relu(model.conv32_3(h))
		
		return h

	def calc_loss(self, x, t,layer,train=True):
		self.clear()

		#print(x.data.shape)
		h = F.relu(model.conv3_32(x))
		h = F.relu(model.conv32_3(h))
		h = F.relu(self.norm_ch3(h, test=not train))

		h = F.relu(model.conv3_32(h))
		h = F.relu(model.conv32_3(h))
		h = F.relu(self.norm_ch3(h, test=not train))

		if layer > 0:
			h = F.relu(model.conv3_32(h))
			h = F.relu(model.conv32_3(h))
			h = F.relu(self.norm_ch3(h, test=not train))
		if layer > 1:
			h = F.relu(model.conv3_32(h))
			h = F.relu(model.conv32_3(h))
			h = F.relu(self.norm_ch3(h, test=not train))
		if layer > 2:
			h = F.relu(model.conv3_32(h))
			h = F.relu(model.conv32_3(h))
			
		loss = F.mean_squared_error(h, t)
		return loss


for layer in range(3):

	#Root file
	#Training_PATH= "/home/ys/Share/7_DL_model_set/ver20170123/15B24/DL_Ans_half"

	Ans_PATH= "/home/ys/Share/7_DL_model_set/ver20170123/15B24/DL_Ans_half"
	Training_PATH= "/home/ys/Share/7_DL_model_set/ver20170123/15B24/trn_half"
	Result_PATH= "/home/ys/Share/15_NN_calc_result/170130_"+str(layer)+"/"
	model_name ="170130.chainermodel"
	if os.path.isdir(Result_PATH)==False:
		os.mkdir(Result_PATH)

	### Read answer image
	Ansfiles = os.listdir(Ans_PATH)

	# 学習対象のモデル作成
	model = Conv()

	chainer.cuda.get_device(0).use()  # Make a specified GPU current
	model.to_gpu()  # Copy the model to the GPU

	#train_image = chainer.Variable(np.asarray([[cv2.imread("ans_area/"+random.choice(Ansfiles),0)/255.0]], dtype=np.float32))


		
	# 最適化の設定
	optimizer = optimizers.Adam()
	optimizer.setup(model)

	# 学習

	#for filename in range(100):
	#	train_image = chainer.Variable(np.asarray([[cv2.imread("denoised/"+Ansfiles[filename], 0)/255.0]], dtype=np.float32))
	#	target = chainer.Variable(np.asarray([[cv2.imread("ans/"+Ansfiles[filename], 0)/255.0]], dtype=np.float32))

	start = time.time()

	for seq in range(2500):
		filenames= random.sample(Ansfiles,100)
		for filename in filenames:
			#print(filename)
			train_image = cv2.imread(Training_PATH+"/"+filename, cv2.IMREAD_COLOR)
			train_image = train_image.transpose(2,0,1)
			train_image = chainer.Variable(cuda.cupy.asarray([(train_image)/255.0], dtype=np.float32))

			target = cv2.imread(Ans_PATH+"/"+filename, cv2.IMREAD_COLOR)
			target = chainer.Variable(cuda.cupy.asarray([target.transpose(2,0,1)/255.0], dtype=np.float32))

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
				#train_image = chainer.Variable(cuda.cupy.asarray([[cv2.imread(Training_PATH +"/"+filename, 0)/255.0]], dtype=np.float32))
				train_image = cv2.imread(Ans_PATH + "/" + filename, cv2.IMREAD_COLOR)
				train_image = chainer.Variable(cuda.cupy.asarray([train_image.transpose(2, 0, 1) / 255.0], dtype=np.float32))
				#print(train_image.data.shape)
				trained = model.forward(train_image,layer).data[0][0]*255
				cv2.imwrite(Result_PATH+str(seq)+"/"+filename, cuda.to_cpu(trained))
				#cuda.to_cpu(trained).shape
			chainer.serializers.save_hdf5(Result_PATH+str(seq)+"/"+model_name, model)
	#	print(model.conv1.W.data[0][0])
	#	trained = model.forward(train_image).data[0][0]*255
	#	cv2.imwrite("trained.jpg", trained)

	# 学習結果の表示
	#print(model.conv1.W.data[0][0])

	if os.path.isdir(Result_PATH+str(seq))==False:
		os.mkdir(Result_PATH+str(seq))
		
	for filename in Ansfiles:
		#train_image = chainer.Variable(cuda.cupy.asarray([[cv2.imread(Training_PATH +"/"+filename, 0)/255.0]], dtype=np.float32))
		train_image = cv2.imread(Ans_PATH + "/" + filename, cv2.IMREAD_COLOR)
		train_image = chainer.Variable(cuda.cupy.asarray([train_image.transpose(2, 0, 1) / 255.0], dtype=np.float32))
		trained = model.forward(train_image,layer).data[0][0]*255
		cv2.imwrite(Result_PATH+str(seq)+"/"+filename, cuda.to_cpu(trained))
	
	chainer.serializers.save_hdf5(Result_PATH+str(seq)+"/"+model_name, model)
