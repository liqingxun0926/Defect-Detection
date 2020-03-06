import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import glob
import time

WIDTH = 64
HEIGHT = 64
CHANNELS = 3

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

def eval(file_dir,pb_file_path, batch_size):
    with tf.Graph().as_default():
        output_graph_def = tf.compat.v1.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            out_softmax = sess.graph.get_tensor_by_name("output:0")

            maxstep,img_batch,label_batch = get_image(file_dir,batch_size)
            acc_count = 0
            max_count  = maxstep*batch_size
            start = time.monotonic()
            for i in range(maxstep):

                img_out_softmax = sess.run(out_softmax, feed_dict={input_x:img_batch[i]})
                prediction = np.argmax(img_out_softmax, axis=1)
                for j in range(batch_size):
                    print("predict label: %s    true label : %s"%(prediction[j],label_batch[i][j]))
                    if str(prediction[j]) == str(label_batch[i][j]) : acc_count = acc_count + 1
            
            inference_time = time.monotonic() - start
            print('defect takes %.2f ms' % (inference_time * 1000))
            print("final accuracy is :{:.2f}%".format(100 * acc_count/max_count))

if __name__ == '__main__':
    eval('./eval/','./eval_logs/frozen_model.pb',12)
