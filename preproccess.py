# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 22:09:54 2018

@author: PC028

samplepre：数据预处理与准备
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os 
import utils as utils


'''定义切割数据体参数'''
s_size = [5, 5]              
Half_size = 2


inline_size = 103
xline_size = 102
t_size = 101


'''读取地震数据，读取断层标签'''
currentpath = os.getcwd()
path = currentpath + '/data'
filename = 'seisclean'
rawseisdata = utils.importfile(path,filename)

filename = 'seiscleanlable'
rawseislabel = utils.importfile(path,filename)


'''选择随机剖面作为样本'''
selected_rate = 0.75
selected_section, unselected_section = utils.selectSection(inline_size, selected_rate)
np.save(path + "/unselected_section.npy",unselected_section)



'''获取训练样本（数据+标签）'''
train_pos = []
seislabel = []
xline_range = np.arange(xline_size) 
xline_range = xline_range[Half_size:xline_size-Half_size]#98

t_range = np.arange(t_size) 
t_range = t_range[Half_size:t_size-Half_size]#97

for xline in xline_range:
    for t in t_range:
        for inline in selected_section:#selected_section=77
            train_pos.append([inline,xline,t])
            seislabel.append(rawseislabel[inline,xline,t])



'''按样本坐标从地震数据体中切割得到样本'''
seisdata = utils.dataMake(train_pos, rawseisdata, Half_size)#731962个样本


''' 调整样本比例 '''  
train_pos = np.array(train_pos)
seislabel = np.array(seislabel)
seisdata, seislabel, train_pos = utils.adjustRate(seisdata, seislabel, train_pos)

 
'''准备数据格式'''
seisdata = np.array(seisdata)
seislabel = np.array(seislabel)
train_pos = np.array(train_pos)
seisdata = np.reshape(seisdata, [-1,5,5,1])
seislabel = np.reshape(seislabel,[-1,1])


'''从样本池里随机按比例选取 训练/验证 样本'''
validation_rate = 0.2
pos_train, pos_validation, x_train, x_validation, y_train, y_validation = utils.randomsplit(train_pos, seisdata, seislabel, validation_rate)


'''对 y_train, y_validation onehot 编码'''
enc = OneHotEncoder() 
enc.fit(y_train)  
y_train = enc.transform(y_train).toarray()  
y_validation = enc.transform(y_validation).toarray() 


'''存储样本数据'''
np.save(path + "/x_train.npy",x_train)
np.save(path + "/x_validation.npy",x_validation)
np.save(path + "/y_train.npy",y_train)
np.save(path + "/y_validation.npy",y_validation)


