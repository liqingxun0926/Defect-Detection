
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "edgetpu.h"
#include "model_units.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

const std::string model_path = "/usr/local/DNN/NJU-AI/defect_detection_v_1_0_2/eval_logs/mymodel2.tflite";

int main(){
    std::unique_ptr<tflite::FlatBufferModel> model =tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    std::unique_ptr<tflite::Interpreter> model_interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
    std::cout<<"hello world!\n"<<std::endl;
}