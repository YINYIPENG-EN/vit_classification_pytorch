import argparse
from loguru import logger

if __name__ == "__main__":
    parse = argparse.ArgumentParser('vit classification')
    parse.add_argument('--model', type=str, default='vit', help='model choose')
    parse.add_argument('--weight', type=str, default='weights/vit-patch_16.pth', help='weight path')
    parse.add_argument('--predict', action='store_true', default=False, help='image predict')
    parse.add_argument('--input_shape', type=list, default=[224, 224], help='model input shape')
    parse.add_argument('--cuda', action='store_true', default=True, help='use cuda')

    # train
    parse.add_argument('--train', action='store_true', default=False, help='train')
    parse.add_argument('--Freeze_Train', action='store_true', default=True, help='Freeze_Train')
    parse.add_argument('--annotation_path', type=str, default="cls_train.txt", help='annotation file path')
    parse.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parse.add_argument('--batch_size', type=int, default=4, help='batch size')
    parse.add_argument('--lr', type=float, default=1e-3, help='Freeze lr')
    parse.add_argument('--UnFreeze_lr', type=float, default=1e-4, help='UnFreeze lr')
    parse.add_argument('--Init_Epoch', type=int, default=0, help='Init_Epoch')
    parse.add_argument('--Freeze_Epoch', type=int, default=50, help='Freeze_Epoch')
    parse.add_argument('--Epochs', type=int, default=100, help='epochs')

    # test
    parse.add_argument('--eval_top1', action='store_true', default=False, help='eval model top1')
    parse.add_argument('--eval_top5', action='store_true', default=False, help='eval model top5')
    opt = parse.parse_args()
    logger.info(opt)
    if opt.predict:
        from tools.predict import image_predict
        image_predict(opt)
    if opt.train:
        from tools.train import train
        train(opt)
    if opt.eval_top1:
        from tools.eval_top1 import evaluteTop1, top1_Classification
        with open(opt.annotation_path, "r") as f:
            lines = f.readlines()
        classfication = top1_Classification(opt)
        top1 = evaluteTop1(classfication, lines)
        print("top-1 accuracy = %.2f%%" % (top1 * 100))
    if opt.eval_top5:
        from tools.eval_top5 import evaluteTop5, top5_Classification

        with open(opt.annotation_path, "r") as f:
            lines = f.readlines()
        classfication = top5_Classification(opt)
        top5 = evaluteTop5(classfication, lines)
        print("top-5 accuracy = %.2f%%" % (top5 * 100))



