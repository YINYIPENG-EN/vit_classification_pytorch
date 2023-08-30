import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
import numpy as np
import torch
from torch import nn
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)
from nets.vit import vit


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
# --------------------------------------------#
class Classification(object):
    _defaults = {
        "classes_path": 'weights/ImageNet.txt'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化classification
    # ---------------------------------------------------#
    def __init__(self, opt, **kwargs):
        self.opt = opt
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   获得种类
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()

    def generate(self):
        if self.opt.model == 'vit':
            self.model = vit(input_shape=self.opt.input_shape, num_classes=self.num_classes, pretrained=False)
        device = torch.device('cuda' if self.opt.cuda else 'cpu')
        self.model.load_state_dict(torch.load(self.opt.weight, map_location=device))
        self.model = self.model.eval()
        print('{} model, and classes loaded.'.format(self.opt.weight))

        if self.opt.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对图片进行不失真的resize
        # ---------------------------------------------------#
        image_data = letterbox_image(image, [self.opt.input_shape[1], self.opt.input_shape[0]])
        # ---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        # ---------------------------------------------------------#
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data)
            if self.opt.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测  model(photo).shape = （batch_size,num_classes）
            # ---------------------------------------------------#

            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        # ---------------------------------------------------#
        #   获得所属种类
        # ---------------------------------------------------#
        class_name = self.class_names[np.argmax(preds)]  # np.argmax(preds)获得preds中概率最大的索引
        probability = np.max(preds)  # 获得概率值

        # ---------------------------------------------------#
        #   绘图并写字
        # ---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        # plt.axis('off')  # 关闭坐标轴
        plt.title('Class:%s Probability:%.3f' % (class_name, probability))
        plt.show()
        return class_name
