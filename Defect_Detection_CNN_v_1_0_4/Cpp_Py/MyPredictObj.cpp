#include "MyPredictObj.h"

std::string Alfeim::MyPredictObj::pyFunc(const char* modelname,const char* funcname,std::initializer_list<std::string> &lst){
        auto &m = ModelMap::get_obj();
        char *tmp = nullptr;
        PyObject *pModule;
        try{
                pModule = m.load_Py_Model(modelname);
                
                if(!pModule) 
                    throw "cannot find such model!";
                std::cout << "[NOTICE] get Python module succeed!" << std::endl;

                PyObject * pFunc = PyObject_GetAttrString(pModule, funcname);
                if (!pFunc || !PyCallable_Check(pFunc)) 
                    throw "cannot find such function!";
                    
                int args_size = lst.size();
                PyObject* args = PyTuple_New(args_size);                
                int pos = 0;
                for(auto i : lst){
                    Alfeim::setargs(args,i,pos++);
                }

                PyObject* pRet = PyObject_CallObject(pFunc, args);    
                PyArg_Parse(pRet,"s",&tmp);
                std::string res(tmp);

                return res;
        }catch(const char* msg){
                std::cerr<<msg<<std::endl;
                exit(-1);
        }

        return "";
}