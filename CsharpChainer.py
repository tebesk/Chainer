#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://kivantium.hateblo.jp/entry/2016/02/04/213050
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
from chainer import cuda, Variable, optimizers
import time
import cupy
from PIL import Image
# 引数の処理
parser = argparse.ArgumentParser(
    description='train convolution filters')
parser.add_argument('org', help='Path to original image')


# クラスの定義
class Conv(chainer.Chain):
    def __init__(self):
        super(Conv, self).__init__(
            # 入力・出力1ch, ksize=3
            conv3_32=F.Convolution2D(3, 32, 5, pad=2),  # conv1=F.Convolution2D(1, 32, 3, pad=1),
            conv32_3=F.Convolution2D(32, 3, 5, pad=2),
            conv3=F.Convolution2D(3, 64, 5, pad=2),  # conv1=F.Convolution2D(1, 32, 3, pad=1),
            conv4=F.Convolution2D(64, 3, 5, pad=2),
            conv7=F.Convolution2D(3, 64, 3, pad=1),  # conv1=F.Convolution2D(1, 32, 3, pad=1),
            conv8=F.Convolution2D(64, 3, 3, pad=1),
            norm_ch3=L.BatchNormalization(3),
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x, train=True):
        self.clear()

        # print(x.data.shape)
        h = F.relu(model.conv3_32(x))
        h = F.relu(model.conv32_3(h))
        h = F.relu(self.norm_ch3(h, test=not train))

        h = F.relu(model.conv3_32(h))
        h = F.relu(model.conv32_3(h))
        h = F.relu(self.norm_ch3(h, test=not train))

        '''
        #if layer > 0:
        h = F.relu(model.conv3_32(h))
        h = F.relu(model.conv32_3(h))
        h = F.relu(self.norm_ch3(h, test=not train))
        #if layer > 1:
        h = F.relu(model.conv3_32(h))
        h = F.relu(model.conv32_3(h))
        h = F.relu(self.norm_ch3(h, test=not train))

        #if layer > 2:
        h = F.relu(model.conv3_32(h))
        h = F.relu(model.conv32_3(h))
        '''
        return h


# Root file
# Training_PATH= "/home/ys/Share/7_DL_model_set/ver20170123/15B24/DL_Ans_half"
img_PATH = r"C:\Users\Sakamoto\Source\Repos\XY2RT\XY2RY\SoundFlow\bin\Debug\beatjudgement"
filename = r"17.bmp"
Result_PATH = r"C:\Users\Sakamoto\Source\Repos\XY2RT\XY2RY\SoundFlow\bin\Debug\DLresult"
model_name = r"C:\Users\Sakamoto\Source\Repos\XY2RT\XY2RY\SoundFlow\bin\Debug\170130_0.chainermodel"

# 学習対象のモデル作成
model = Conv()
chainer.serializers.load_hdf5(model_name,model)
#chainer.cuda.get_device(0).use()  # Make a specified GPU current
model.to_cpu()  # Copy the model to the GPU

cv_image = cv2.imread(img_PATH + r"\\"+ filename, cv2.IMREAD_COLOR)
print (cv_image.shape)
## 最後のtranseposeは RGBRGBRGB ----> RRRRGGGGBBBって感じに変換
train_image = chainer.Variable(np.asarray([cv_image.transpose(2, 0, 1) / 255.0], dtype=np.float32))
trained = model.forward(train_image)#.data[0][0] * 255
'''
print ("train_image.size")
print (train_image.size)
print ("trained.size")
print (trained.size)
print ("trained.data.size")
print (trained.data.size)
#test = trained.reshape(1,3,300,256)
print ("modi trained.size")
print (trained.size)
'''
im = np.zeros((300,256,3))
im[:,:,0] = trained.data[0][0]*255
im[:,:,1] = trained.data[0][1]*255
im[:,:,2] = trained.data[0][2]*255
resized_img = cv2.resize(im, (256,300))
#im =im.transpose(0,2,3,1)
#Image.fromarray(im).save(Result_PATH +  r"\\"+ filename)
cv2.imwrite(Result_PATH +  r"\\"+ filename, resized_img)