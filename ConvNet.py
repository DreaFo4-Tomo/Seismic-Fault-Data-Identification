# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:04:23 2018

@author: cp

ConvNet：用于断层识别的卷积神经网络
"""

LEARNING_RATE = 0.00001
BATCH_SIZE = 32


import numpy as np
import tensorflow as tf
from xavier import xavier


class ConvSegNet(object):
    
    def __init__(self):
        self.model_path = 'mymodel'

        self.x = tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name = 'x')
        self.y = tf.placeholder(tf.float32, shape=[None, 2], name = 'y')
        self.batch_size = BATCH_SIZE
        
#        self.input_size = tf.placeholder(tf.int8,shape=[1], name = 'input_size')
#        self.input_size = self.x.shape[0].value
        
        self.weights = self._initialize_weights()        
        self.layers = self._initial_layers()

        
    
    def rectraining(self, x_train, x_validation, y_train, y_validation, total_epochs):
        m_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.layers['cnn']['fc2']))   
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(m_loss)      

        correct_pred = tf.equal(tf.argmax(self.layers['cnn']['fc2'],1),tf.argmax(self.y,1))        
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
        res_pre = tf.cast(tf.argmax(tf.nn.softmax(self.layers['cnn']['fc2']),1), tf.float32, name = 'prediction')  
        allloss = []
        allacc = []
        saver=tf.train.Saver()               
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())                
            n_samples = len(x_train)
            steps = int(n_samples/self.batch_size)
            display_step = 1000
            disorder = np.linspace(0,n_samples-1,n_samples,dtype = int)
            for j in range(total_epochs):
                print("epoch:",j,"start")  
                np.random.shuffle(disorder)
                index = np.linspace(0,self.batch_size-1,self.batch_size,dtype = int)  
                for i in range(steps):
                    disorder_index = disorder[index]
                    data =  x_train[disorder_index]
                    label =  y_train[disorder_index]
                    index = index + self.batch_size 
                    loss, opt = sess.run([m_loss, optimizer], feed_dict={self.x: data, self.y: label})                                     
                    if i % display_step == 0:
                        print("step:", i, "--", steps, "loss=","{:.9f}".format(loss))
                        allloss.append(loss)
                batch_x_validation = x_validation[0:1000]
                batch_y_validation = y_validation[0:1000]
                acc = sess.run([accuracy], feed_dict={self.x: batch_x_validation, self.y: batch_y_validation})         
                print (("accuracy = "), acc)
                allacc.append(acc)
            print(" Rectraining Optimization Finished!!")  
            saver.save(sess,self.model_path + '/model')
            sess.close()
            return allloss,allacc

    '''
    设定网络结构
    '''    
    def _initial_layers(self):
        cnn = dict()
        cnn['h1_conv'] = tf.nn.relu(tf.nn.conv2d(self.x, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + self.weights['b1'])
        #h1_conv.shape=(None, 5, 5, 5, 32)
        cnn['h1_pool'] = tf.nn.max_pool(cnn['h1_conv'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #h1_pool.shape=(None, 3, 3, 3, 32)
        cnn['h2_conv'] = tf.nn.relu(tf.nn.conv2d(cnn['h1_pool'], self.weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + self.weights['b2'])
        #h2_conv.shape = (None, 3, 3, 3, 64)
        cnn['h2_pool'] = tf.nn.max_pool(cnn['h2_conv'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    
        #h2_pool.shape = (None, 2, 2, 2, 64)
        cnn['fc_add'] = tf.nn.relu(tf.nn.conv2d(cnn['h2_pool'], self.weights['fc_w'], strides=[1, 1, 1, 1], padding='VALID') + self.weights['fc_b'])
        #h4_add.shape = (None,1,1,256)  ???
        cnn['fc_reshape'] = tf.reshape(cnn['fc_add'],[-1, 256])
        #h4_reshape.shape = (None,256)  ???
        cnn['fc1'] = tf.nn.relu(tf.matmul(cnn['fc_reshape'], self.weights['fc_w1']) + self.weights['fc_b1'])
        cnn['fc1_drop'] = tf.nn.dropout(cnn['fc1'], 0.5)
        #h5_dropout =  tf.nn.dropout(h5, keep_prob = 0.5)
        cnn['fc2']  = (tf.matmul(cnn['fc1_drop'], self.weights['fc_w2']) + self.weights['fc_b2'])
 
        layers = dict()
        layers['cnn'] = cnn
        return layers

    '''
    设定网络层参数
    '''
    def _initialize_weights(self):
        weights = dict()
        xa = xavier()
        weights['w1'] = tf.Variable(xa.conv2d([3, 3,  1, 32], constant = 1), name = 'w1') 
        weights['b1'] = tf.Variable(tf.zeros([32],dtype=tf.float32), name = 'b1')
        weights['w2'] = tf.Variable(xa.conv2d([3, 3,  32, 64], constant = 1), name = 'w2') 
        weights['b2'] = tf.Variable(tf.zeros([64],dtype=tf.float32), name = 'b2')
        weights['fc_w'] = tf.Variable(xa.conv2d([2, 2,  64, 256], constant = 1), name = 'fc_w') 
        weights['fc_b'] = tf.Variable(tf.zeros([256],dtype=tf.float32), name = 'fc_b')
        weights['fc_w1'] = tf.Variable(xa.nn([256, 512], constant = 1), name = 'fc_w1') 
        weights['fc_b1'] = tf.Variable(tf.zeros([512],dtype=tf.float32), name = 'fc_b1')
        weights['fc_w2'] = tf.Variable(xa.nn([512, 2], constant = 1), name = 'fc_w2') 
        weights['fc_b2'] = tf.Variable(tf.zeros([2],dtype=tf.float32), name = 'fc_b2')
        return weights
        
    def getWeights(self,name):
        sess = tf.Session()
        return sess.run(self.weights[name])
    
    def getShape(self, name1, name2):
        return self.layers[name1][name2].shape
    
          
