下面是文件夹的结构，train里面有两个文件夹，分别是缺陷件和非缺陷件，尺寸为64 * 64
```
├── eval(存放测试图片)
├── eval_logs(存放测试用的静态模型)
├── train
│   └── defect
│   └── no_defect
├── train_logs(存放训练临时生成的cpkt文件)
├── input_data.py(输入模块，用于读取数据) 
├── MyNet.py(训练脚本) 在使用时如果增加 --EVAL/-e 参数,显示输入1时即进入测试模式，格式如"python MyNet.py --EVAL(or -e) 1".默认情况下如果不输入参数则进入训练模式
└── test_on_pb.py(测试脚本,请在包含tensorflow的虚拟环境下测试)  无参数
└── test_on_tflite.py(测试脚本,请在包含tensorflow lite的虚拟环境下测试)  输入参数名(arg)对应在GPU下还是在TPU下推断，格式为 "python test_on_tflite.py --DEVICE arg"
```

建议在tensorflow2.0环境下进行测试,在tensorflow1.X环境下进行训练、验证并生成量化模型

add on 2019/11/26 本项目可直接在tensorflow1.15下训练并测试

