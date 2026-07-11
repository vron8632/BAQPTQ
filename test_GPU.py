# -*- coding:utf-8 -*-
# 开发人员 : csu·pan-_-||
# 开发时间 : 2025-10-03 7:38
# 文件名称 : test_GPU.py
# 开发工具 : PyCharm
# 功能描述 : 自己修改

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU arch: {torch.cuda.get_device_capability()}")

import torch
print(f"Current GPU arch: {torch.cuda.get_device_capability()}")

from mmcv import ops
print(ops.get_compiling_cuda_version())
