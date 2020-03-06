from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
import collections
import argparse
import input_data


#注意  要把训练 和 验证的 pb模型分开存储（为了后续量化）
#所以我们实际需要的模型存放在VAL_LOGS_DIR下


BATCH_SIZE = 12                                      #batch的大小
MAX_STEP = 4000                                     #总的训练步数
FILEPATH = './train/'                                 #存放训练图片的位置
VALPATH = './eval/'
TRAIN_LOGS_DIR = './train_logs/'      #存放训练模型
VAL_LOGS_DIR = './eval_logs/'
WIDTH= 64                                                   #图像压缩后的宽
HEIGHT= 64                                                 #图像压缩后的高
CHANNELS=3                                              #图像通道数，RGB下为3
LEARNING_RATE = 0.0001                     #学习率
ratio = 0.8                                                     #训练集与验证集的比率，一般为82开所以设置为0.8

   #定义网络一些层的接口函数
def weight_variable(shape, stddev,name="weights"):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(val,shape, name="biases"):
    initial = tf.constant(value=val, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(input, w,strides):
    return tf.nn.conv2d(input, w, strides = strides, padding='SAME')

def pool_max(input,ksize,strides,padding,name):
    return tf.nn.max_pool(input,
                               ksize=ksize,
                               strides=strides,
                               padding=padding,
                               name=name)

def bn_layer(x,is_training):
    return tf.layers.batch_normalization(x ,training=is_training)

def fc(input, w, b):
    return tf.matmul(input, w) + b


def build_network(input,IF_TRAIN):
     # 卷积层1        64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('layer1_conv') as scope:
        kernel = weight_variable([3, 3, 3, 64],1.0)
        biases = bias_variable(0.1,[64])
        tmpconv =  conv2d(input, kernel,[1,1,1,1])+ biases
        if IF_TRAIN:
            bn1 = bn_layer(tmpconv,True)
        else:
            bn1 = bn_layer(tmpconv,False)
        conv1 = tf.nn.relu(bn1, name='conv2d_out')

    # 池化层1       3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
    with tf.variable_scope('layer_pool_lrn') as scope:
        pool1 = pool_max(conv1,[1,3,3,1],[1,2,2,1],'SAME','pooling')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_out')

    # 卷积层2        16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('layer2_conv') as scope:
        kernel = weight_variable([3, 3, 64, 16],0.1)
        biases = bias_variable(0.1,[16])
        tmpconv = conv2d(norm1, kernel,[1,1,1,1]) + biases
        if IF_TRAIN:
            bn2 = bn_layer(tmpconv,True)
        else:
             bn2 = bn_layer(tmpconv,False)
        conv2 = tf.nn.relu(bn2, name='conv2d_out')

    # 池化层2       3x3最大池化，步长strides为2，池化后执行lrn()操作，
    with tf.variable_scope('layer2_pool_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
        pool2 = pool_max(norm2,[1,3,3,1],[1,1,1,1],'SAME','pooling')

    # 全连接层3
    with tf.variable_scope('layer3_fullyconnect') as scope:
        shape = int(np.prod(pool2.get_shape()[1:]))
        kernel = weight_variable([shape,128],0.005)
        biases = bias_variable(0.1,[128])
        flat = tf.reshape(pool2,[-1,shape])
        fc3 = tf.nn.relu(fc(flat,kernel,biases),name = 'fullyconnect_out')


    # 全连接层4
    with tf.variable_scope('layer4_fullyconnect') as scope:
        kernel = weight_variable([128,128],0.005)
        biases = bias_variable(0.1,[128])
        fc4 = tf.nn.relu(fc(fc3,kernel,biases),name = 'fullyconnect_out')

    with tf.variable_scope('layer5_fullyconnect') as scope:
        kernel = weight_variable([128,2],0.005)
        biases = bias_variable(0.1,[2])

    softmax_linear = tf.add(tf.matmul(fc4, kernel), biases, name='output')
    #计算损失函数及反向传播

    return softmax_linear




def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def train_net():
    #------进入计算图--------
    x_train,y_train,x_val,y_val = input_data.read_img(FILEPATH,WIDTH,HEIGHT,CHANNELS,ratio)
    x_train_batch,y_train_batch = input_data.bulid_batch(x_train,y_train,BATCH_SIZE)
    x_val_batch,y_val_batch = input_data.bulid_batch(x_val,y_val,BATCH_SIZE)
    batch_train_len = x_train_batch.shape[0]
    batch_val_len = x_val_batch.shape[0]

   #定义网络    x为输入占位符      y为输出占位符
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE,HEIGHT, WIDTH, CHANNELS], name='input')
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels_placeholder')
    #flag= tf.placeholder(tf.bool, [],name = "is_training")
    
    softmax_linear = build_network(x,True)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=softmax_linear, labels=y, name='xentropy_per_example')
    train_loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=2000) 
    train_step = trainning(train_loss,LEARNING_RATE)

    #准确率计算
    correct = tf.nn.in_top_k(softmax_linear, y, 1)
    correct = tf.cast(correct, tf.float16)
    train_acc = tf.reduce_mean(correct)

    #------------结束计算图-------------

    with tf.Session() as sess:

        saver = tf.compat.v1.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        valstep = 0

        #训练
        try:
            ckpt = tf.train.get_checkpoint_state(TRAIN_LOGS_DIR)
            global_step = 0
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
                    
            for i in range(MAX_STEP):
                
                #if_train = True
                pos = i % batch_train_len
                _,acc,loss = sess.run([train_step,train_acc,train_loss],
                         feed_dict={x : x_train_batch[pos], y : y_train_batch[pos]})

                #每50步打印一次准确率和损失函数
                if  i% 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(i, loss, acc*100.0))

                #每200步用验证集的数据进行验证
                if i%200 == 0:
                    #if_train = False    #量化模式下用变量替代占位符.注意 如果要用tflite的话,if_train不要用占位符！
                    vpos = valstep % batch_val_len
                    val_loss, val_acc = sess.run([train_loss, train_acc],
                                                 feed_dict={x : x_val_batch[vpos], y : y_val_batch[vpos]})

                    valstep = valstep+1
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(i, val_loss, val_acc*100.0))

                #每500步保存一次变量值
                if i%500 == 0:
                    checkpoint_path = os.path.join(TRAIN_LOGS_DIR, 'saved_model.ckpt')
                    tmpstep = i + int(global_step)
                    saver.save(sess, checkpoint_path, global_step=tmpstep)
                  
        
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)





#IF_TRAIN = Fasle时，用于测试

def get_image(file_dir,batch_size):
    #先载入图片
    imgs   = []
    labels = []
    
    for im in glob.glob(file_dir + '/*.jpg'):
        #print('reading the image: %s' % (im))
        img = io.imread(im)
        img = transform.resize(img, (WIDTH, HEIGHT, CHANNELS))
        imgs.append(img)
        if 'qx' in im:
            labels.append(0)
        else:
            labels.append(1)

    imgs = np.asarray(imgs,np.float32)
    labels = np.asarray(labels,np.int32)

    #然后生成batch
    image_batch = []
    label_batch = []
    
    border = imgs.shape[0]

    if border % batch_size != 0:
        border = border - border%batch_size

    maxstep = border//batch_size
    for i in range(maxstep):
        label_batch.append(labels[i*batch_size:i*batch_size + batch_size])
        image_batch.append(imgs[i*batch_size:i*batch_size + batch_size])

    return maxstep,np.asarray(image_batch),np.asarray(label_batch)


def eval():
    maxstep,image_batch,label_batch = get_image(VALPATH,BATCH_SIZE)
    with tf.Graph().as_default():
        input_x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,WIDTH,HEIGHT,CHANNELS],name = "input")
        logit = build_network(input_x,False)
        tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(TRAIN_LOGS_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            acc_count = 0
            max_count = maxstep*BATCH_SIZE

            for i in range(maxstep):
                output = sess.run(logit,feed_dict = {input_x: image_batch[i]})
                prediction = np.argmax(output, axis=1)
                for j in range(BATCH_SIZE):
                    print("predict label: %s    true label : %s"%(prediction[j],label_batch[i][j]))
                    if str(prediction[j]) == str(label_batch[i][j]) : acc_count = acc_count + 1
        
            print("final accuracy is :{:.2f}%".format(100 * acc_count/max_count))
            graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            tf.train.write_graph(graph,VAL_LOGS_DIR,'frozen_model.pb',as_text=False)

           

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '-e','--EVAL', required=False,default=0,
                      help='choose to train or eval',type=int)
    parser.add_argument( '-b','--BATCH', required=False,default=12,
                      help='set batch_size',type=int)
    parser.add_argument( '-s','--STEP', required=False,default=4000,
                      help='set maxstep',type=int)
    parser.add_argument( '-i','--IMGPATH', required=False,default='./train/' ,
                      help='set train image path',type=str)
    parser.add_argument( '-v','--VALPATH', required=False,default='./eval/' ,
                      help='set val image path',type=str)
    parser.add_argument( '-tl','--TRAINLOGS', required=False,default='./train_logs/' ,
                      help='set train logs path',type=str) 
    parser.add_argument( '-vl','--VALLOGS', required=False,default='./eval_logs/' ,
                      help='set val logs path',type=str)
    parser.add_argument( '-wd','--WIDTH', required=False,default=64 ,
                      help='set image width',type=int)
    parser.add_argument( '-ht','--HEIGHT', required=False,default=64 ,
                      help='set image height',type=int)
    parser.add_argument( '-ch','--CHANNELS', required=False,default=3 ,
                      help='set image channels',type=int)
    parser.add_argument( '-lr','--LRATE', required=False,default=0.0001 ,
                      help='set learning rate',type=float)
    parser.add_argument( '-rt','--RATIO', required=False,default=0.8 ,
                      help='set the ratio of train/train+val',type=float)
    
    args = parser.parse_args()
    
    flag = args.EVAL
    BATCH_SIZE = args.BATCH                                    
    MAX_STEP = args.STEP                                   
    FILEPATH = args.IMGPATH                                 
    VALPATH = args.VALPATH
    TRAIN_LOGS_DIR = args.TRAINLOGS     
    VAL_LOGS_DIR = args.VALLOGS
    WIDTH= args.WIDTH                                                 
    HEIGHT= args.HEIGHT                                               
    CHANNELS=args.CHANNELS                                     
    LEARNING_RATE = args.LRATE                   
    ratio = args.RATIO       

    if flag:   
        eval()
    else:
        train_net()
