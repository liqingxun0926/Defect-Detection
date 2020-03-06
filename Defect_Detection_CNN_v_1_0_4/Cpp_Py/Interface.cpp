#include "Interface.h"
#include <string>

//the definition of initialize
void Alfeim::initialize(){
    Py_Initialize();
    PyRun_SimpleString("import sys");  
    PyRun_SimpleString("sys.path.append('./')");  
}

//the definition of setargs
void Alfeim::setargs(PyObject* args, const std::string &str,int pos){
    int split = str.find("_");
    std::string type = str.substr(0,split);
    std::string context = str.substr(split + 1);
    if(type == "i"){
        PyTuple_SetItem(args, pos, Py_BuildValue("i", std::stoi(context)));
    }else if(type == "d"){
        PyTuple_SetItem(args, pos, Py_BuildValue("d", std::stod(context)));
    }else if(type == "s"){
        PyTuple_SetItem(args, pos, Py_BuildValue("s",  context.c_str()));
    }
}

//design my own res handler
void Alfeim::handle_res_1(const std::string &res){
    std::vector<std::vector<int>> batch_res;
    int len = res.size();
    int last = 0;

    for(int i = 0 ; i < len ; ){
            last = i;
            std::vector<int> tmp;
            for(; i < len && i < last + 12 ; ++i ){
                tmp.push_back(res[i] - '0');
            }
            batch_res.push_back(tmp);
    }

    int batch_num = 0;
    for(auto i : batch_res){
        std::cout << "batch num [" << batch_num <<"] :";
        for(auto j : i){                                                       
            std:std::cout << (j == 0 ? "  defective  " : " indefective ");
        }
        std::cout<<std::endl;
        batch_num++;
    }

}

//the definition of ModelMap::get_obj() 
Alfeim::ModelMap& Alfeim::ModelMap::get_obj(){
    static ModelMap m;
    return m;
}

//the definition of ModelMap::load_Py_Model
PyObject* Alfeim::ModelMap:: load_Py_Model(const std::string modelname){
    if(!models.count(modelname)) models[modelname] = PyImport_ImportModule(modelname.c_str());
    return models[modelname];
}



