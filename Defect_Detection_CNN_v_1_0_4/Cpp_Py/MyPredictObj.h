#ifndef _MY_PREDICT_H_
#define _MY_PREDICT_H_
#include "Interface.h"

namespace Alfeim{
    //the basic predictobj offered by alfeim
    //you can declare  your own predictobj in this header file and  define it in MyPredictObj.cpp
    //please declare the function as override to remember overriding the virtual function
    class MyPredictObj:public PredictObj{
        public:
            MyPredictObj()=default;
            ~MyPredictObj()=default;
            std::string pyFunc(const char* modelname,const char* funcname,std::initializer_list<std::string> &lst) override;
    };
}

#endif