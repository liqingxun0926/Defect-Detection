import argparse
#import tflite_runtime.interpreter as tflite #if use tenosorflow lite then use it to replace import tensorflow as tf
import tensorflow as tf
import tensorflow.compat.v2.lite as tflite
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import glob
import time

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'  #edgetpu的库

WIDTH = 64
HEIGHT = 64
CHANNELS = 3


#生成解释器
def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.experimental.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


#载入图片
def get_image(file_dir,batch_size):
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


def eval(file_dir,tflite_file_path, batch_size):
    result = ""
    #输入参数
    parser = argparse.ArgumentParser(
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--DEVICE', required=False,default='TPU',
                      help='choose to use TPU or GPU ')
    args = parser.parse_args()
    DEVICE = str(args.DEVICE) 

    #创建解释器,根据输入参数选择GPU或者TPU
    if DEVICE == 'GPU':
        interpreter = tflite.Interpreter(model_path=tflite_file_path)
    elif DEVICE == 'TPU':
        interpreter = make_interpreter(tflite_file_path)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    #读取输入输出层
    output_details = interpreter.get_output_details()
    maxstep,image_batch,label_batch = get_image(file_dir,batch_size)
    print(input_details)
    print(output_details)

    #推断循环
    acc_count = 0
    max_count = maxstep * batch_size
    start = time.monotonic()
    for i in range(maxstep):
        #载入图片并推断
        interpreter.set_tensor(input_details[0]['index'],image_batch[i])
        interpreter.invoke()
        #输出并预测
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output, axis=1)
        for j in range(batch_size):
            print("predict label: %s    true label : %s"%(prediction[j],label_batch[i][j]))
            result += str(prediction[j])
            if str(prediction[j]) == str(label_batch[i][j]) : acc_count = acc_count + 1
    
    inference_time = time.monotonic() - start
    print('detect %d pictures takes %.2f ms' % (max_count,inference_time * 1000))
    print("final accuracy is :{:.2f}%".format(100 * acc_count/max_count))
    return result


if __name__ == '__main__':
    eval('./eval/','./eval_logs/mymodel2.tflite',12)


    
