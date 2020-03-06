#ifndef _INTERFACE_H_
#define _INTERFACE_H_
#include <Python.h>
#include<iostream>
#include<memory>
#include<vector>
#include<initializer_list>
#include<unordered_map>

//this is the header file which define the interface of some basic class and function
//you can add more interface in namespace Alfeim or your own namespace
//do not change the args of these functions below

namespace Alfeim{    

    //function initialize should be called in the beggining
    //if you want to call python function in cpp,remember initialize the python module first
    //remeberimport current path
    void initialize();

    //function setargs is used in setting python function args
    //args:The python args tuple
    //str:The args of function in the python model,which is saved as string
    //pos:The position of the arg
    void setargs(PyObject* args, const std::string &str,int pos);

    //the function to handle the result
    void handle_res_1(const std::string &res);

    class ModelMap{
        //this class is designed to saving the python model
        //the class is designed as singleton,which means that it cannot be construct by the constructor,so the constructor is declared as private
        //and so the copy construct function and  the copy assignment function is declared as delete
        //the only way to get the object of this class is to call use get_obj(),which should be called such as : auto &p = ModelMap::get_obj();
        //the hashmap is designed to save the python model,which key stands for the model name(filename),and the value stands for the PyObject pointer
        public:
            ~ModelMap(){};
            
            ModelMap(const ModelMap&) = delete;
            
            ModelMap& operator=(const ModelMap&) = delete;

            static ModelMap& get_obj();

            PyObject* load_Py_Model(const std::string modelname);

        private:
            ModelMap(){};
            std::unordered_map<std::string,PyObject*> models;
        
    };
    
    class PredictObj{
        //this class is the interface of prediction object
        //you can designed your own predict class  by  inheritancing this basic class in public
        //remember to overwrite the virtual function pyFunc to designed your own python function caller
        public:
            PredictObj()=default;
            ~PredictObj()=default;
            virtual std::string pyFunc(const char* modelname,const char* funcname,std::initializer_list<std::string> &lst) = 0 ;
    };

}

#endif // DEBUG