#ifndef EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"


// Builds tflite Interpreter capable of running Edge TPU model.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context);


#endif  // EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_