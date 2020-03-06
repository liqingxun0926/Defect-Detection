cmake_minimum_required(VERSION 3.5)
project(tensorflow_lite_cpp)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

# Setup Tensorflow C++
set(TENSORFLOW_INCLUDES
        /usr/local/include/tf/
        /usr/local/include/tf/bazel-genfiles
        /usr/local/include/tf/tensorflow/
        /usr/local/include/tf/tensorflow/third-party
        /usr/local/include/eigen3
	/usr/local/include/tf/downloads/absl
	/usr/local/include/tf/downloads/flatbuffers/include/)
set(TENSORFLOW_LIBS
	/usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0
        /usr/local/lib/libtensorflow_cc.so
        /usr/local/lib/libtensorflow_framework.so)

include_directories(
        ${TENSORFLOW_INCLUDES}
        ${PROJECT_SOURCE_DIR}/third_party/eigen3
)

# CNN Net C++
add_executable(main main.cpp)
target_link_libraries(main
            ${TENSORFLOW_LIBS}
            ${OpenCV_LIBS}
	    )

