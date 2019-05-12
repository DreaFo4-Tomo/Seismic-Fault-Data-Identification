# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:38:47 2019

@author: PC028

tfcnn：运行神经网络
"""


'''tensorflow搭建训练网络'''
import numpy as np
from ConvNet import ConvSegNet
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''读取地震数据并扩容，读取断层标签'''
currentpath = os.getcwd()
path = currentpath + '/data'

x_train = np.load(path + "/x_train.npy")
y_train = np.load(path + "/y_train.npy")
x_validation = np.load(path + "/x_validation.npy")
y_validation = np.load(path + "/y_validation.npy")

'''清空原来的tf图模型'''
tf.reset_default_graph() 


''' 神经网络训练 '''
epoch_num = 15
net = ConvSegNet()
allloss,allacc = net.rectraining(x_train, x_validation, y_train, y_validation,epoch_num)
len = len(allloss)

fig1 = plt.figure()
plt.plot(np.arange(len),allloss)
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('the trend of loss')

fig2 = plt.figure()
plt.plot(np.arange(epoch_num),allacc)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('the trend of accuracy')
plt.show()