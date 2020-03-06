# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:32:04 2020

@author: 40490
"""

import os
import shutil

output_train_path = '/home/a/Datasets/cat&dog/class/cat'
output_test_path = '/home/a/Datasets/cat&dog/class/dog'

if not os.path.exists(output_train_path):
    os.makedirs(output_train_path)
if not os.path.exists(output_test_path):
    os.makedirs(output_test_path)

def scanDir_lable_File(dir,flag = True):

    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)
    if not os.path.exists(output_test_path):
        os.makedirs(output_test_path)
    for root, dirs, files in os.walk(dir, True, None, False):  # 遍列目录
        # 处理该文件夹下所有文件:
        for f in files:
            if os.path.isfile(os.path.join(root, f)):
                a = os.path.splitext(f)
                # print(a)
                # label = a[0].split('.')[1]
                label = a[0].split('.')[0]
                print(label)
                if label == 'cat':
                    img_path = os.path.join(root, f)
                    mycopyfile(img_path, os.path.join(output_train_path, f))
                else:
                    img_path = os.path.join(root, f)
                    mycopyfile(img_path, os.path.join(output_test_path, f))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))


root_path = '/home/a/Datasets/cat&dog'
train_path = root_path+'/train/'
test_path = root_path+'/test/'
scanDir_lable_File(train_path)