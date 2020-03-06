cmake_minimum_required(VERSION 3.5)
project(Cpp_Py_Test)

# Setup Tensorflow C++
set(PYTHON_INCLUDES
	/home/alfeim/anaconda3/envs/tensorflow2.0/include/python3.6m/)
set(PYTHON_LIBS
	/home/alfeim/anaconda3/envs/tensorflow2.0/lib/libpython3.6m.so)

include_directories(
        ${PYTHON_INCLUDES}
)

# CNN Net C++
add_executable(main main.cpp Interface.cpp MyPredictObj.cpp)
target_link_libraries(main
            ${PYTHON_LIBS}
	    )

