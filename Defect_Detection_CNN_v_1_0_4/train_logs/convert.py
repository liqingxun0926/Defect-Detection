import tensorflow as tf

in_path = "./saved_model.pb"
out_path = "./mymodel2.tflite"
# out_path = "./model/quantize_frozen_graph.tflite"

# 模型输入节点
input_tensor_name = ["input"]
input_tensor_shape = {"input":[12,64,64,3]}
# 模型输出节点
classes_tensor_name = ["output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
                                            input_tensor_name, classes_tensor_name,
                                            input_shapes = input_tensor_shape)
converter.post_training_quantize = False
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)
