# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:09:56 2018

@author: PC028

xavier：参数均一化初始化器
"""


''' 功能：xavier初始化器，把权重初始化在low和high范围内(满足N(0,2/Nin+Nout)) '''
import numpy as np
import tensorflow as tf

class xavier(object):
    '''
    for DNN
    '''
    def nn(self, shape, constant=1):
        fan_in = shape[0]
        fan_out = shape[1]
        low = -constant * np.sqrt(6.0/(fan_in+fan_out))
        high = constant * np.sqrt(6.0/(fan_in+fan_out))
        return tf.random_uniform(shape,minval=low,maxval=high,dtype=tf.float32)
    
    
    '''
    输入(for conv2d):
        shape = [] of this filter, 4 dim
        fan_in = input_channels = self[2]
        fan_out = output_channels = self[3]
    输出(for 2Dcov):
        初始化权重，shape = []
    '''
    def conv2d(self, shape, constant = 1):
        fan_in = shape[2]
        fan_out = shape[3]
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)


    '''
    输入(for conv3d):
        shape = [] of this filter, 5 dim
        fan_in = input_channels = self[3]
        fan_out = output_channels = self[4] 
    输出(for 3Dcov):
        初始化权重，shape = []
    '''
    def conv3d(self, shape, constant = 1):
        fan_in = shape[3]
        fan_out = shape[4]
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)
    
    

