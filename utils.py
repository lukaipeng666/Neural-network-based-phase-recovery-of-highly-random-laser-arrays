import time

import torchvision
from tqdm import tqdm
import torch
import os

global mean_lsG_2


def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss):
    global mean_lsG_2
    pd = tqdm(train_loader)
    loss_D, loss_G = 0, 0
    loss_G_1 = 0
    step = 0
    G.train()
    D.train()
    for idx, data in enumerate(pd):
        in_img = data[0].to(device)
        real_img = data[1][0].to(device)
        real_img_1 = data[1][1].to(device)
        # 先训练D
        fake_img = G(in_img)[0]
        D_fake_out = D(fake_img.detach(), in_img).squeeze()
        D_real_out = D(real_img, in_img).squeeze()
        ls_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda())
        ls_D2 = loss(D_real_out, torch.ones(D_real_out.size()).cuda())
        ls_D = (ls_D1 + ls_D2) * 0.5

        optim_D.zero_grad()
        ls_D.backward()
        optim_D.step()

        # 再训练G
        fake_img_0 = G(in_img)
        fake_img = fake_img_0[0]
        fake_img_1 = fake_img_0[1]
        D_fake_out = D(fake_img, in_img).squeeze()
        ls_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda())
        ls_G2 = l1_loss(fake_img, real_img)
        ls_G2_1 = l1_loss(fake_img_1, real_img_1) * 100
        loss_G_1 += ls_G2_1
        ls_G = ls_G1 + ls_G2 * 100 + ls_G2_1

        optim_G.zero_grad()
        ls_G.backward()
        optim_G.step()

        loss_D += ls_D
        loss_G += ls_G

        pd.desc = 'train_{} G_0_loss: {} G_1_loss: {} D_loss: {}'.format(epoch, ls_G.item(), ls_G2_1.item(),
                                                                         ls_D.item())
        # 绘制训练结果
        # if idx % plot_every == 0:
        #     writer.add_images(tag='train_epoch_{}'.format(epoch), img_tensor=0.5 * (fake_img + 1), global_step=step)
        #     step += 1
    mean_lsG = loss_G / len(train_loader)
    mean_lsD = loss_D / len(train_loader)
    mean_lsG_2 = loss_G_1 / len(train_loader)
    return mean_lsG, mean_lsD, mean_lsG_2


@torch.no_grad()
def val(G, D, val_loader, loss, device, l1_loss, epoch):
    global best_image
    pd = tqdm(val_loader)
    loss_D, loss_G = 0, 0
    loss_G_1 = 0
    G.eval()
    D.eval()
    all_loss = 10000
    for idx, item in enumerate(pd):
        in_img = item[0].to(device)
        real_img = item[1][0].to(device)
        real_img_1 = item[1][1].to(device)
        fake_img_0 = G(in_img)
        fake_img = fake_img_0[0]
        fake_img_1 = fake_img_0[1]
        D_fake_out = D(fake_img, in_img).squeeze()
        ls_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda())
        ls_D = ls_D1 * 0.5
        ls_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda())
        ls_G2 = l1_loss(fake_img, real_img)
        ls_G2_1 = l1_loss(fake_img_1, real_img_1) * 100
        ls_G = ls_G1 + ls_G2 * 100 + ls_G2_1
        loss_G_1 += ls_G2_1
        loss_G += ls_G
        loss_D += ls_D
        pd.desc = 'val_{}: G_0_loss: {} G_1_loss: {} D_loss: {}'.format(epoch, ls_G.item(), ls_G2_1.item(),
                                                         ls_D.item())
        loss_G_mean = loss_G_1 / len(val_loader)
        if loss_G_mean > mean_lsG_2 + 1:
            time.sleep(120)

        # 保存最好的结果
        all_ls = ls_G + ls_D
        if (epoch - 1) % 10 == 0:
            if all_ls < all_loss:
                all_loss = all_ls
                best_image = fake_img_1
            result_img = (best_image + 1) * 0.5
            if not os.path.exists('./results'):
                os.mkdir('./results')

            torchvision.utils.save_image(result_img, './results/val_epoch{}.jpg'.format(epoch))
