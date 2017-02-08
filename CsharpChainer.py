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
import numpy as np
import multiprocessing
from multiprocessing import Queue

from chainer import cuda, Variable, optimizers
import time
import cupy
from PIL import Image
import shutil
import Utility
# 引数の処理
parser = argparse.ArgumentParser(
    description='train convolution filters')
parser.add_argument('org', help='Path to original image')


# クラスの定義
class Conv(chainer.Chain):
    # def __init__(self):# コンストラクタ
    def __init__(self, Path, ResultPath):  # コンストラクタ
        self.Path = Path
        self.ResultPath = ResultPath
        self.queue = Queue()
        self.list = []

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

    '''------------------------------------
    Network Construction
    ------------------------------------'''
    def forward(self, x, train=True):
        self.clear()

        # print(x.data.shape)
        h = F.relu(self.conv3_32(x))
        h = F.relu(self.conv32_3(h))
        h = F.relu(self.norm_ch3(h, test=not train))

        h = F.relu(self.conv3_32(h))
        h = F.relu(self.conv32_3(h))
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

    '''------------------------------------
    Network processing
    ------------------------------------'''
    def RunNetwork(self, filename, Path, ResultPath):

        cv_image = cv2.imread(Path  + r"\\" + filename, cv2.IMREAD_COLOR)
        orgHeight, orgWidth = cv_image .shape[:2]
        cv_image = cv2.resize(cv_image, (orgHeight//2, orgWidth//2))

        ## 最後のtranseposeは RGBRGBRGB ----> RRRRGGGGBBBって感じに変換
        train_image = chainer.Variable(np.asarray([cv_image.transpose(2, 0, 1) / 255.0], dtype=np.float32))
        trained = self.forward(train_image)  # calculate network

        im = np.zeros((300, 256, 3))
        im[:, :, 0] = trained.data[0][0] * 255
        im[:, :, 1] = trained.data[0][1] * 255
        im[:, :, 2] = trained.data[0][2] * 255
        resized_img = cv2.resize(im, (256, 300))

        cv2.imwrite(ResultPath + r"\\" + filename, resized_img)

    '''------------------------------------
    multithread processing -Read directory-
    ------------------------------------'''
    def Step1(self):
        i = 0
        print("start1 started i = ", str(i))
        os.mkdir(self.Path+ r"\\tmp_for_dl")

        while True:
            filelist = Utility.check_file(self.Path, ".bmp")

            if len(filelist) != 0:#if there is no file in the directory
                for file in filelist:
                    if os.path.exists(self.Path + r"\\" + file):
                        shutil.move(self.Path+ r"\\" + file, self.Path+ r"\\tmp_for_dl")
                        self.queue.put(file)

            else:
                i = i + 1
                print("step1 i=", str(i))
                if i > 10:
                    print("step1 break")
                    break

    '''------------------------------------
    multithread processing -Calculate Deep Learning-
    ------------------------------------'''
    def Step2(self):
        i = 0
        print("step2 started")
        start = time.time()

        while True:
            if self.queue.empty():
                print("step2 no queue")
                i = i + 1
                if i == 10:
                    print("step2 break")
                    os.rmdir(self.Path+ r"\\tmp_for_dl")
                    break
            else:
                print("step2:", self.queue.qsize())
                tmp_file = self.queue.get()
                if os.path.exists(self.Path+ r"\\tmp_for_dl\\" + tmp_file):
                    #run Neural network
                    self.RunNetwork(tmp_file, self.Path+ r"\\tmp_for_dl", self.ResultPath)
                    os.remove(self.Path+ r"\\tmp_for_dl\\" + tmp_file)
                elapsed_time = time.time() - start
                print(("run NN elapsed_time:{0}".format(elapsed_time)), "[sec]")



if __name__ == "__main__":
    model_name = r"C:\Users\Sakamoto\Source\Repos\XY2RT\XY2RY\SoundFlow\bin\Debug\170130_0.chainermodel"
    Path = r"C:\Users\Sakamoto\PycharmProjects\Chainer\sample"
    Result_PATH = r"C:\Users\Sakamoto\PycharmProjects\Chainer\sample2"
    start = time.time()
    # 既存のモデルを用いて、ネットワーク拡大
    model = Conv(Path, Result_PATH)
    chainer.serializers.load_hdf5(model_name, model)
    model.to_cpu() # Copy the model to the CPU

    elapsed_time = time.time() - start
    print(("elapsed_time:{0}".format(elapsed_time)) , "[sec]")

    t1 = multiprocessing.Process(target=model.Step1)
    t2 = multiprocessing.Process(target=model.Step2)

    t1.start()
    t2.start()

    elapsed_time = time.time() - start
    print (("elapsed_time:{0}".format(elapsed_time)) , "[sec]")