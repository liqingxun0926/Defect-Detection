# This is a Makefile to cross-compile minimal.cc example.
# 1. Download TensorFlow to Linux machine:
#    $ git clone https://github.com/tensorflow/tensorflow.git
# 2. Download external dependencies for TensorFlow Lite:
#    $ tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
# 3. Cross-compile TensorFlow Lite for aarch64:
#    $ tensorflow/tensorflow/lite/tools/make/build_generic_aarch64_lib.sh
# 4. Cross-compile minimal.cc example for aarch64:
#    $ TENSORFLOW_DIR=<location> make
# 5. Copy the following files to Coral Dev board:
#      * Generated 'minimal' binary
#      * Model file 'mobilenet_v1_1.0_224_quant_edgetpu.tflite' from 'test_data'
#      * Image file 'resized_cat.bmp' from 'test_data'
#    and finally run 'minimal' binary on the board:
#    $ minimal mobilenet_v1_1.0_224_quant_edgetpu.tflite resized_cat.bmp
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TENSORFLOW_DIR = '/usr/local/include/tf/'

main: main.cpp
	g++ -std=c++11 -o main main.cpp \
	-I$(MAKEFILE_DIR) \
	-I$(TENSORFLOW_DIR) \
	-I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-l:libedgetpu.so.1.0 -lpthread -lm -ldl

clean:
	rm -f main
