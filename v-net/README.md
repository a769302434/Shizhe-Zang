## 3DUNet implemented with pytorch

## Introduction
参考代码： [project](https://github.com/panxiaobai/lits_pytorch).
原论文: [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf)

### Requirements:  
```angular2
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```
### Start:  

把nii格式的数据放在dataset里，然后先运行process.py，对数据预处理，再运行train.py