import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes, weights_init
from nets.vit import vit
from utils.utils_fit import fit_one_epoch


def train(opt):
    classes_path = 'weights/cls_classes.txt'
    pretrained = False
    val_split = 0.1
    class_names, num_classes = get_classes(classes_path)
    if opt.model == 'vit':
        model = vit(num_classes=num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if opt.weight != '':
        print('Loading {} into state dict...'.format(opt.weight))
        device = torch.device('cuda' if opt.cuda else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.weight, map_location=device)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict.keys() == model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model_train = model.train()
    if opt.cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    loss_history = LossHistory("logs/")

    with open(opt.annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines) * val_split)
    num_train   = len(lines) - num_val

    if True:
        # ----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        # ----------------------------------------------------#
        lr = opt.lr
        Init_Epoch = opt.Init_Epoch
        Freeze_Epoch = opt.Freeze_Epoch

        epoch_step = num_train // opt.batch_size
        epoch_step_val = num_val // opt.batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DataGenerator(lines[:num_train], opt.input_shape, True)
        val_dataset = DataGenerator(lines[num_train:], opt.input_shape, False)
        gen = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)
        gen_val = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True,
                             drop_last=True, collate_fn=detection_collate)
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if opt.Freeze_Train:
            model.freeze_backbone()

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Freeze_Epoch, opt.Cuda)
            lr_scheduler.step()

    if True:
        # ----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        # ----------------------------------------------------#
        lr = opt.UnFreeze_lr
        Batch_size = opt.batch_size
        Freeze_Epoch = opt.Freeze_Epoch
        Epoch = opt.Epochs

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DataGenerator(lines[:num_train], opt.input_shape, True)
        val_dataset = DataGenerator(lines[num_train:], opt.input_shape, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=opt.num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=opt.num_workers, pin_memory=True,
                             drop_last=True, collate_fn=detection_collate)
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if opt.Freeze_Train:
            model.Unfreeze_backbone()

        for epoch in range(Freeze_Epoch, Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Epoch, opt.Cuda)
            lr_scheduler.step()


