import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PolypFormer import PolypFormer
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging

import matplotlib.pyplot as plt


def structure_loss(pred, mask):   # 计算权重，其中mask是输入的二值掩码。使用平均池化操作来平滑mask，然后计算其与原始mask之间的差异，并加上一个偏置项。
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='None')     # 计算加权二元交叉熵损失。pred 是模型的预测值，
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))              # mask 是目标掩码。reduce='none' 参数表示不对损失进行降维。

    pred = torch.sigmoid(pred)              # 对模型的预测值应用sigmoid激活函数，将其缩放到(0, 1)范围内，以得到概率值。
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()             # 返回加权二元交叉熵损失和加权IoU损失的平均值作为结构化损失。


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()                                            # 将模型设置为评估模式，这会关闭dropout和batch normalization
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)    # 创建测试数据加载器，用于加载图像和标签，并进行预处理。
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()           # 从测试数据加载器中加载图像、标签和名称
        gt = np.asarray(gt, np.float32)                     # 将标签转换为 NumPy 数组，并指定数据类型为 np.float32
        gt /= (gt.max() + 1e-8)                             # 对标签进行归一化
        image = image.cuda()                                # 将图像转移到GPU上

        res = model(image)                                  # 用模型对图像进行推理，得到预测结果。
        # eval Dice
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()    # 将预测结果转换为概率值，并将其从 GPU 移回 CPU，并且去除维度为 1 的尺寸。
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))                # 将预测结果与标签展平为一维数组
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)           # 计算预测结果和标签的交集
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice                                    # 累加Dice系数到总和中
    return DSC / num1                                       # 返回Dice相似度系数的平均值


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()                                           # 将模型设置为训练模式。
    global best
    size_rates = [0.75, 1, 1.25]                            # 定义图像尺寸缩放比例。
    loss_P_record = AvgMeter()                              # 初始化用于记录损失的平均计量器。
    for i, pack in enumerate(train_loader, start=1):        # 迭代训练数据加载器
        for rate in size_rates:                             # 对每个尺寸缩放比例进行迭代。
            optimizer.zero_grad()                           # 清除优化器的梯度。
            # ---- data prepare ----
            images, gts = pack                              # 从数据包中获取图像和标签。
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----                             # 从数据包中获取图像和标签。
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            #print(images.size())
            P= model(images)                               # 通过模型进行前向传播，得到预测结果P2
            # ---- loss function ----
            # loss_P1 = structure_loss(P1, gts)
            loss_P = structure_loss(P, gts)                # 计算结构化损失。
            loss = loss_P
            # ---- backward ----
            loss.backward()                                 # 反向传播计算梯度。
            clip_gradient(optimizer, opt.clip)              # 裁剪梯度，防止梯度爆炸。
            optimizer.step()                                # 更新模型参数
            # ---- recording loss ----
            if rate == 1:
                loss_P_record.update(loss_P.data, opt.batchsize)

        # ---- train visualization  每迭代20步，或总步数打印一次----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P_record.show()))            # 记录每20步的损失值
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + str(epoch) + 'PolypPVT.pth')
    # choose the best model

    global dict_plot                # 声明了一个全局变量dict_plot，用于存储模型在不同数据集上的测试结果。
    test1path = './dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:        # 每个epoch结束后遍历五个数据集
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            #------------调用test函数进行模型在当前数据集上的测试，并返回测试结果（即平均dice值）
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

            meandice = test(model, test_path, 'test')       # 对测试集进行测试，得到平均 dice 值。
            dict_plot['test'].append(meandice)
            if meandice > best:
                best = meandice
                torch.save(model.state_dict(), save_path + 'PolypFormer.pth')
                torch.save(model.state_dict(), save_path + str(epoch) + 'PolypFormer-best.pth')
                print('##############################################################################best', best)
                logging.info('##############################################################################best:{}'.format(best))


def plot_train(dict_plot=None, name=None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        plt.axhline(color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()                    # 显示图例
    plt.savefig('eval.png')
    plt.show()
    
    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}     #注意这里是自己数据val里面的列表
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'PolypFormer'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    parser.add_argument('--resume', default='', help='resume from checkpoint')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = PolypFormer().cuda()
    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    
    # plot the eval.png in the training stage
    plot_train(dict_plot, name)
