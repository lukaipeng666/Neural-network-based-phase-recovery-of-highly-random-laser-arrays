from torch.utils.tensorboard import SummaryWriter
import argparse
from Data_set import CreateDatasets
from Data_set_2 import CreateDatasets2
from split_my_data import split_data
import os
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from net import U_net, unet_D
import time
import threading
from net import downsample

torch.autograd.set_detect_anomaly(True)
from utils import train_one_epoch, val

from data_generate import data_generate_seed

global train_datasets_real, val_datasets_real, train_loader, val_loader, train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out


def train(opt):
    global val_datasets_real, train_datasets_real, generate_just
    generate_just = 0
    print_every = opt.every
    batch = opt.batch
    data_path = opt.dataPath
    output_path = opt.output_path
    device = torch.device('cuda')
    epochs = opt.epoch
    img_size = opt.imgsize

    if not os.path.exists(opt.savePath):
        os.mkdir(opt.savePath)
    train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out = data_generate_seed()
    train_datasets_real = CreateDatasets2(train_ori_imglist, train_ori_imglist_out)
    val_datasets_real = CreateDatasets2(val_ori_imglist, val_ori_imglist_out)

    train_loader = DataLoader(dataset=train_datasets_real, batch_size=batch, shuffle=True,
                              num_workers=opt.numworker,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_datasets_real, batch_size=batch, shuffle=True,
                            num_workers=opt.numworker,
                            drop_last=True)

    # 实例化网络
    model = U_net()
    pix_G = model.to(device)
    pix_D = unet_D().to(device)

    # for name, module in pix_G.named_children():
    #     if isinstance(module, downsample):
    #         for param in module.parameters():
    #             param.requires_grad = False  # 冻结住下采样层
    # for name, param in model.named_parameters():
    #     print(f'{name}:requires_grad={param.requires_grad}')
    # 定义优化器和损失函数
    optim_G = optim.Adam(filter(lambda p: p.requires_grad, pix_G.parameters()), lr=0.0002, betas=(0.1, 0.999))
    optim_D = optim.Adam(filter(lambda p: p.requires_grad, pix_D.parameters()), lr=0.0002, betas=(0.1, 0.999))

    loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    criterion = nn.L1Loss()
    epoch = 0

    # 加载预训练权重
    if opt.weight != '':
        ckpt = torch.load(opt.weight)
        pix_G.load_state_dict(ckpt['G_model'], strict=False)
        pix_D.load_state_dict(ckpt['D_model'], strict=False)
        epoch = 0

    writer = SummaryWriter(log_dir='train_logs')
    # 开始训练
    loss_mean = 1
    while loss_mean > 0.30:

        def data_generation_thread():
            global train_datasets_real, val_datasets_real, train_loader, val_loaderm, train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out, generate_just
            train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out = data_generate_seed()
            train_datasets_real = CreateDatasets2(train_ori_imglist, train_ori_imglist_out)
            val_datasets_real = CreateDatasets2(val_ori_imglist, val_ori_imglist_out)
            generate_just = 2

        if generate_just == 0:
            generate_just = 1
            thread = threading.Thread(target=data_generation_thread)
            thread.start()

        if generate_just == 2:
            train_loader = DataLoader(dataset=train_datasets_real, batch_size=batch, shuffle=True,
                                      num_workers=opt.numworker,
                                      drop_last=True)
            val_loader = DataLoader(dataset=val_datasets_real, batch_size=batch, shuffle=True,
                                    num_workers=opt.numworker,
                                    drop_last=True)
            generate_just = 0
            print('updated')
            time.sleep(0.1)
        epoch += 1
        train_one_epoch(G=pix_G, D=pix_D, train_loader=train_loader,
                        optim_G=optim_G, optim_D=optim_D, writer=writer, loss=loss, device=device,
                        plot_every=print_every, epoch=epoch, l1_loss=l1_loss)

        val(G=pix_G, D=pix_D, val_loader=val_loader, loss=loss, l1_loss=l1_loss, device=device, epoch=epoch)

        if epoch % 20 == 0:
            torch.save({
                'G_model': pix_G.state_dict(),
                'D_model': pix_D.state_dict(),
                'epoch': epoch
            }, './weights/Bi_Ushape_Attendion.pth')


def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=4)
    parse.add_argument('--epoch', type=int, default=1000)
    parse.add_argument('--imgsize', type=int, default=256)
    parse.add_argument('--dataPath', type=str, default='./input', help='data root path')
    parse.add_argument('--output_path', type=str, default='./output', help='data root path')
    parse.add_argument('--weight', type=str, default='./weights/Bi_Ushape_Attendion.pth', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./weights', help='weight save path')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=8, help='plot train result every * iters')
    opt = parse.parse_args()
    return opt


if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)
