# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:08:56 2019

@author: PC028

utils：工具包
"""

''' 功能：导入mat数据格式文件 '''
import numpy as np
import scipy.io as sio

def importfile(path, name):
    dictname = name.replace('.mat','')
    rawdata = sio.loadmat(path + '/' +  name)[dictname]
    return rawdata

    
'''
功能：制作样本数据
Half_size: (尺寸-1)/2, dtype=int 
pos:[inline,xline,t]
'''
def dataMake(pos, seisdata, Half_size):

    data = []
    length_size = 2*Half_size + 1
    length = np.arange(length_size)
    length = length - Half_size
    image = np.zeros([length_size,length_size])

    for dot in pos:
        inline = dot[0]
        xline = dot[1]
        t = dot[2]
        for i in length:
            for j in length:
                    image[i,j] = seisdata[inline, xline+j,t+i]#image为inline面上的切割体[inline, xline,t]为中心点
        copysite = image.copy()
        data.append(copysite)   
    return np.array(data)



'''选择随机剖面作为样本'''
def selectSection(inline_size, selected_rate):
    #select_dim = 0;                 #选择的剖面：0--inline，1--xline, 2--time；多用inline
    select_section = np.arange(inline_size)
    np.random.shuffle(select_section)
    selected_section = select_section[0:int(selected_rate*select_section.size)]
    unselected_section = select_section[int(selected_rate*select_section.size):inline_size]
    return selected_section, unselected_section



''' 调整样本比例 '''    
def adjustRate(seisdata, seislabel, train_pos):
    
    Pos_index = np.where(seislabel>0.5)
    Neg_index = np.where(seislabel<0.5)
    Positive_example = len(Pos_index[0])
    Negative_example = len(Neg_index[0])
    increase_rate = int(Negative_example/Positive_example)


    reseisdata = []
    reseislabel = []
    train_repos = []
    shuffleindex = np.arange(len(seislabel))
    np.random.shuffle(shuffleindex)
    for i in shuffleindex:
        if(seislabel[i] == True):
            for add in range(increase_rate):
                reseisdata.append(seisdata[i])
                reseislabel.append(seislabel[i])
                train_repos.append(train_pos[i])
        else:
            for add in range(1):
                reseisdata.append(seisdata[i])
                reseislabel.append(seislabel[i])
                train_repos.append(train_pos[i])
    print('Positive example number:', Positive_example, '\t\t Negative example number:', Negative_example)
    print('After adjusting posi:nega rate:  ' , Positive_example*increase_rate/Negative_example)

    return np.array(reseisdata),  np.array(reseislabel),  np.array(train_repos)


'''
功能：随机按占比选取训练集与验证集,已经随机打乱

x：输入训练数据
y：输出标签数据
test_size：测试集占比
random_state：随机种子点
'''
def randomsplit(pos, x, y, test_size, random_state = None):
    from sklearn.model_selection import train_test_split
    pos_train, pos_validation, x_train, x_validation, y_train, y_validation = train_test_split(pos, x, y, test_size = test_size, random_state = random_state)
    return pos_train, pos_validation, x_train, x_validation, y_train, y_validation
