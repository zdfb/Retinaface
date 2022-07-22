# Retinaface
基于Pytorch实现的Retinaface，复现了论文中的使用多尺度测试方式。
## 性能效果
| Backbone| easy | medium | hard |
|:-|:-:|:-:|:-:|
| Resnet152 | 96.20% | 95.08% | 90.30% |

## 预训练模型
+ 基于Resnet152的人脸检测模型Retinaface。<br>
>- 链接: https://pan.baidu.com/s/1OkobTSFZUYGqEw9ecc5wGQ
>- 提取码：xp1d

## 训练
### 1. 下载数据集并放置在Data下
>- 链接: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS

### 2. 开始训练
``` bash
python train.py
```
## 测试图片
修改utils/utils_yoloface.py文件中的model_path指向训练好的模型。
在predict.py文件下输入图片路径，运行：
``` bash
python predict.py
```
## widerface 数据集测试
修改utils/utils_yoloface.py文件中的model_path指向训练好的模型。
``` bash
python test_widerface.py
```
``` bash
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
## Reference
- https://github.com/biubug6/Pytorch_Retinaface

