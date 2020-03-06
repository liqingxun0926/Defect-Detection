import os
import glob
from skimage import io, transform
import numpy as np

def read_img(path,WIDTH,HEIGHT,CHANNELS,ratio):
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (WIDTH, HEIGHT, CHANNELS))
            imgs.append(img)
            labels.append(idx)

    #转换为array
    imgs = np.asarray(imgs,np.float32)
    labels = np.asarray(labels,np.int32)
    num_example = imgs.shape[0]
    s = np.int(num_example * ratio)

    #打乱乱序
    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    np.random.shuffle(labels)
   

    #x_train 训练集图片  y_train 训练集标签
    #x_val 验证集图片  y_al 验证集标签

    x_train = imgs[:s]
    y_train = labels[:s]
    x_val   = imgs[s:]
    y_val   = labels[s:]

    return x_train,y_train,x_val,y_val

def bulid_batch(image, label, batch_size):
    
    #生成batch 多幅图片为一个batch
    image_batch = []
    label_batch = []
    
    border = image.shape[0]

    if border % batch_size != 0:
        border = border - border%batch_size

    maxstep = border//batch_size
    for i in range(maxstep):
        label_batch.append(label[i*batch_size:i*batch_size + batch_size])
        image_batch.append(image[i*batch_size:i*batch_size + batch_size])

    return np.asarray(image_batch),np.asarray(label_batch)
