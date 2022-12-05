运行指令记得要加上环境变量
XLA_PYTHON_CLIENT_ALLOCATOR=platform

蒸馏训练
XLA_PYTHON_CLIENT_ALLOCATOR=platform CUDA_VISIBLE_DEVICES=0,1,2,3 python distillation_train.py

在orginal model 进行采样 8192步
CUDA_VISIBLE_DEVICES=0,1 python original_sample.py

若分开采样
则
CUDA_VISIBLE_DEVICES=0 python original_sample0.py
CUDA_VISIBLE_DEVICES=1 python original_sample1.py
CUDA_VISIBLE_DEVICES=2 python original_sample2.py
CUDA_VISIBLE_DEVICES=3 python original_sample3.py
