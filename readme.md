# 环境

torch >=1.2.0

# 参数说明

--model:vit模型

--weight:权重路径

--predict：开启预测功能

--input_shape: 网络输入大小，默认为224×224

--cuda:是否采用GPU推理

--train:训练模型

--Freeze_Train:冻结训练

--annotation_path：标签txt路径

--num_workers: number of workers

--batch_size:batch size

--lr:冻结训练的学习率大小，默认1e-3

--UnFreeze_lr:解冻训练的学习率大小

--Init_Epoch:初始epoch

--Freeze_Epoch:冻结训练epoch,默认50，表示前50轮冻结训练

--Epochs:总的训练epochs,默认100，表示50~100解冻训练

--eval_top1:top1测试

--eval_top5：top5测试

# 预测

如果是自己训练的数据集和预权重，需要在weights/下新建一个cls_classes.txt，并写入自己类【比如cat,dog】，仿照ImageNet.txt写就行。然后修改classification.py中的classes_path为自己的txt路径，比如：

```python
class Classification(object):
    _defaults = {
        "classes_path": 'weights/cls_classes.txt'
    }
```



```python
python demo.py --predict --weight 【你的权重路径】
```

eg:

```pyhon
python demo.py --predict --weight weights/vit-patch_16.pth
```

# 训练自己的数据集

## 数据集的准备：

将训练数据集和测试数据集放在datasets/train和test中，每个类单独建一个文件，比如训练猫狗数据集。

```shell
datasets/
|-- test
|   |-- cat
|   `-- dog
`-- train
    |-- cat
    `-- dog

```

修改txt_annotation.py中的classes为自己的类。

然后运行txt_annotation.py。

此时会生成cls_train.txt和cls_test.txt文件

## 预权重下载：

该预权重是ImageNet 1000个类的。

链接：https://pan.baidu.com/s/12fjPdGVgXuX7rvQxA_NsQQ 
提取码：yypn

## 训练：

```bash
python demo.py --train --batch_size 8 --weights weight/vit-patch_16.pth
```

每个epoch的训练权重会自动保存在logs中。

可以结合其他参数进行训练

# 测试

## top1测试

```
python demo.py --eval_top1 --weights 【你的权重路径】
```

## top5测试

```
python demo.py --eval_top5 --weights 【你的权重路径】
```

