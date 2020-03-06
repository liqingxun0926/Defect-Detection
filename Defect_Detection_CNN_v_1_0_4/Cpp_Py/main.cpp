#include <iostream>
#include "Interface.h"
#include "MyPredictObj.h"


int main(){
    //remember to initialize python module first!
    Alfeim::initialize();
    //create an python module
    Alfeim::PredictObj   *p = new Alfeim::MyPredictObj{};
    //dsetting the args list,which will be decode and passed on as the ars of python function
    //all the args is saved as string first,to distingush the type of ars,declare the type before symbol '_', 'i' stands for int32,'s' stands for string,'d' stands for double(float) 
    std::initializer_list<std::string> args{"s_./eval/","s_./eval_logs/mymodel2.tflite","i_12"};
    std::string res = p->pyFunc("test_on_tflite","eval",args);
    Alfeim::handle_res_1(res);
    return 0;
}