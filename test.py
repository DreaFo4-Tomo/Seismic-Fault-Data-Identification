# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:19:38 2018

@author: PC028

tftestinline：测试inline方向切片
"""



import numpy as np
import utils as utils
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def rectest(test_x):
    tf.reset_default_graph() 
    with tf.Session() as sess:
        model_path = "mymodel"
        saver = tf.train.import_meta_graph(model_path + '/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("x").outputs[0] 
        pred = graph.get_operation_by_name("prediction").outputs[0]
        pred = sess.run([pred], feed_dict={x: test_x})
        sess.close()
    
    pred = np.array(pred)


    '''调整显示大于0.8的概率'''
    index = np.where(pred>0.80)
    y_threshold = np.zeros(pred.shape)
    y_threshold[index] = 1
    y_threshold = y_threshold.reshape(98,97)

    '''显示0.5概率'''
    index = np.where(pred>0.50)
    y_half = np.zeros(pred.shape)
    y_half[index] = 1
    y_half = y_half.reshape(98,97)
   
    '''显示概率'''
    y_prob = pred.reshape(98,97)
    
    return y_threshold, y_half, y_prob



inline_size = 103
xline_size = 102
t_size = 101
Half_size = 2

currentpath = os.getcwd()
path = currentpath + '/data'
filename = 'seisclean'
rawdata = utils.importfile(path,filename)



#inline = 24
unselected_section = np.load(path + "/unselected_section.npy")

inline = unselected_section[2]
print inline
xline_range = np.arange(xline_size) 
xline_range = xline_range[Half_size:xline_size-Half_size]
t_range = np.arange(t_size) 
t_range = t_range[Half_size:t_size-Half_size]

section = []
section_label = []
dot = np.zeros([3],dtype=int)
for xline in xline_range:
    for t in t_range:
        dot[0] = inline
        dot[1] = xline
        dot[2] = t
        x = dot.copy()
        section.append(x)
section = np.array(section)


filename = 'seiscleanlable'
rawlabel = utils.importfile(path,filename)
section_label = rawlabel[inline,:,:]

testsection = utils.dataMake(section, rawdata, Half_size)
testsection = np.reshape(testsection,[-1,5,5,1])

res_threshold, res_half, res_prob = rectest(testsection)

fig1 = plt.figure()
plt.imshow(np.transpose(rawdata[inline,:,:]))
plt.title('Seismic data profile')
fig2 = plt.figure()
plt.imshow(np.transpose(section_label))
plt.title('Fault data label map')
fig3 = plt.figure()
plt.imshow(np.transpose(res_prob))
plt.title('Fault predict map')
plt.show()

