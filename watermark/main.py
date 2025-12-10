#!/usr/bin/env python
import torch
import os
import torch.nn
import torch.optim
import calculate_PSNR_SSIM
import numpy as np
from model import *
import config as c
from torch.utils.tensorboard import SummaryWriter
import datasets
import viz
import warnings
from util import attack, gauss_noise, mse_loss, computePSNR, dwt, iwt
import torchvision
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前使用设备：{device}")

# 网络参数数量

def load(net, optim, name, load_opt=True):
    if not os.path.exists(name):
        print(f"警告：预训练文件 {name} 不存在，跳过加载，从头训练！")
        return net, optim
    # ===================== 避免CUDA冲突 =====================
    state_dicts = torch.load(name, map_location=device)
    # ==========================================================================
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    if load_opt == True:
        optim.load_state_dict(state_dicts['opt'])
    return net, optim



def embed_attack(net, input_img, attack_method):
    #################
    #    forward:   #
    #################
    output = net(input_img)
    output_container = output.narrow(1, 0, 4 * c.channels_in)
    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
    output_z = gauss_noise(output_z.shape)
    container_img = iwt(output_container)

    #################
    #   attack:   #
    #################
    attack_container = attack(container_img, attack_method)
    input_container = dwt(attack_container)

    return container_img, attack_container, output_z, output_container, input_container

def train_epoch(net, disc, optim_gen, optim_disc, step, attack_method=None, i_epoch=None, writer=None, mode='train', lam=(1.0, 1.0), device='mps'):
    r_loss_list, g_loss_list, pre_loss_list, post_loss_list, psnr_c, psnr_s, total_loss_list = [], [], [], [], [], [], []
    lam_c, lam_s = lam
    if mode != 'train':
        dataloader = datasets.testloader
        net.eval()
    else:
        dataloader = datasets.trainloader
        net.train()
        disc.train()  # 仅在训练模式启用

    for i_batch, data in enumerate(dataloader):
        data = data.to(device)
        num = data.shape[0] // 2
        host = data[:num]  # 原始清晰图像（GAN的真实样本）
        secret = data[num:num * 2]
        host_input = dwt(host)
        secret_input = dwt(secret)

        input_img = torch.cat((host_input, secret_input), 1)
        # 可逆块输出：container_img是需要优化清晰度的生成图像
        steg_img, attack_container, output_z, output_container, input_container = embed_attack(net, input_img, attack_method)
        container_img = steg_img  # GAN生成器的输出

        if step == 1 or step == 2:
            input_container = net.pre_enhance(attack_container)
            input_container = dwt(input_container)

        #################
        # 反向传播准备（原有逻辑）
        #################
        output_rev = torch.cat((input_container, output_z), 1)
        output_image = net(output_rev, rev=True)
        extracted = output_image.narrow(1, 4 * c.channels_in,
                                         output_image.shape[1] - 4 * c.channels_in)
        extracted = iwt(extracted)
        if step == 1 or step == 2:
            extracted = net.post_enhance(extracted)

        #################
        # 原有损失计算
        #################
        c_loss = mse_loss(container_img, host)
        s_loss = mse_loss(extracted, secret)

        extracted = extracted.clip(0, 1)
        secret = secret.clip(0, 1)
        host = host.clip(0, 1)
        container = steg_img.clip(0, 1)

        psnr_temp = computePSNR(extracted, secret)
        psnr_s.append(psnr_temp)
        psnr_temp_c = computePSNR(host, container)
        psnr_c.append(psnr_temp_c)

        #################
        # GAN损失计算（仅在训练模式）
        #################
        if mode == 'train':
            # 1. 训练判别器
            optim_disc.zero_grad()
            # 真实样本损失（宿主图像）
            real_pred = disc(host)
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred) * 0.9)  # 标签平滑
            # 生成样本损失（容器图像）
            fake_pred = disc(container_img.detach())  # 切断生成器梯度
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            # 判别器总损失
            disc_loss = (real_loss + fake_loss) * 0.5
            disc_loss.backward()
            optim_disc.step()

            # 2. 训练生成器（原有网络）
            optim_gen.zero_grad()
            # 原有任务损失
            if step == 1:
                total_loss = s_loss
            elif step == 0 or step == 2:
                total_loss = lam_s * s_loss + lam_c * c_loss
            # GAN生成器损失（欺骗判别器）
            fake_pred_gen = disc(container_img)
            gan_loss = F.binary_cross_entropy(fake_pred_gen, torch.ones_like(fake_pred_gen))
            # 总损失 = 原有损失 + 对抗损失
            total_loss += c.gan_weight * gan_loss
            total_loss.backward()
            optim_gen.step()

        else:
            # 非训练模式仅计算原有损失
            if step == 1:
                total_loss = s_loss
            elif step == 0 or step == 2:
                total_loss = lam_s * s_loss + lam_c * c_loss

        #################
        # 日志与保存（原有逻辑）
        #################
        if mode == 'test':
            torchvision.utils.save_image(host, c.IMAGE_PATH_host + '%.5d.png' % i_batch)
            torchvision.utils.save_image(container, c.IMAGE_PATH_container + '%.5d.png' % i_batch)
            torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i_batch)
            torchvision.utils.save_image(extracted, c.IMAGE_PATH_extracted + '%.5d.png' % i_batch)

        g_loss_list.append(c_loss.item())
        r_loss_list.append(s_loss.item())
        total_loss_list.append(total_loss.item())

    #################
    # 可视化日志（原有逻辑）
    #################
    if mode == 'val':
        before = 'val_'
    else:
        before = ''

    if mode != 'test':
        writer.add_scalars(f"{before}c_loss", {f"{before}guide loss": np.mean(g_loss_list)}, i_epoch)
        writer.add_scalars(f"{before}s_loss", {f"{before}rev loss": np.mean(r_loss_list)}, i_epoch)
        writer.add_scalars(f"{before}PSNR_S", {f"{before}average psnr": np.mean(psnr_s)}, i_epoch)
        writer.add_scalars(f"{before}PSNR_C", {f"{before}average psnr": np.mean(psnr_c)}, i_epoch)
        writer.add_scalars(f"{before}Loss", {f"{before}Loss": np.mean(total_loss_list)}, i_epoch)
        if mode == 'train':
            writer.add_scalar("disc_loss", disc_loss.item(), i_epoch)  # 新增判别器损失日志

    return np.mean(total_loss_list)



def train(net, disc, optim_gen, optim_disc, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer=None, expinfo='', lam=(1.0, 1.0)):
    writer = SummaryWriter(comment=expinfo, filename_suffix="steg")
    # 验证集初始化（不训练GAN）
    val_loss = train_epoch(net, disc, optim_gen, optim_disc, step, attack_method, start_epoch, mode='val', writer=writer, lam=lam)

    for i_epoch in range(start_epoch + 1, end_epoch + 1):
        #################
        # 训练循环（包含GAN）
        #################
        train_loss = train_epoch(net, disc, optim_gen, optim_disc, step, attack_method, i_epoch, writer=writer, lam=lam)

        #################
        # 验证（不训练GAN）
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                val_loss = train_epoch(net, disc, optim_gen, optim_disc, step, attack_method, i_epoch, mode='val', writer=writer, lam=lam)

        info = [np.round(train_loss, 2), np.round(val_loss, 2), np.round(np.log10(optim_gen.param_groups[0]['lr']), 2), attack_method]
        viz.show_loss(visualizer, info)

        #################
        # 模型保存（包含GAN参数）
        #################
        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({
                'opt_gen': optim_gen.state_dict(),
                'opt_disc': optim_disc.state_dict(),
                'net': net.state_dict(),
                'disc': disc.state_dict()
            }, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')

        weight_scheduler.step()

    # 最终保存
    torch.save({
        'opt_gen': optim_gen.state_dict(),
        'opt_disc': optim_disc.state_dict(),
        'net': net.state_dict(),
        'disc': disc.state_dict()
    }, f'final_state/{expinfo}.pt')
    writer.close()


def model_init(step, load_path='', load_opt=True):
    net = PRIS(in_1=3, in_2=3)
    if step == 0:
        lr = c.lr
        for name, para in net.named_parameters():
            if 'inbs' in name:
                para.requires_grad = True
            elif 'enhance' in name:
                para.requires_grad = False

    elif step == 1:
        lr = c.lr
        for name, para in net.named_parameters():
            if 'inbs' in name:
                para.requires_grad = False
            elif 'enhance' in name:
                para.requires_grad = True
    elif step == 2:
        lr = c.lr * 0.1
        for name, para in net.named_parameters():
            if 'inbs' in name:
                para.requires_grad = True
            elif 'enhance' in name:
                para.requires_grad = True


    optim = torch.optim.Adam(filter(lambda x:x.requires_grad, net.parameters()), lr=lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    net.to(device)
    init_model(net)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    if load_path != '':
        net, optim = load(net, optim, load_path, load_opt)
        print(f'load from {load_path}')
    return net, optim, weight_scheduler


def main(attack_method, step, load_path='', start_epoch=0, end_epoch=1600, lam=(1.0, 1.0)):
    warnings.filterwarnings("ignore")
    if step == 0:
        expinfo = f'{attack_method}_hinet_pretrain'
    elif step == 1:
        expinfo = f'{attack_method}_enhance_pretrain'
    elif step == 2:
        expinfo = f'{attack_method}_enhance_finetune'

    # 加载路径处理（原有逻辑）
    if load_path == '':
        load_opt = False
        if step == 1:
            load_path = f'final_state/{attack_method}_hinet_pretrain.pt'
        elif step == 2:
            load_path = f'final_state/{attack_method}_enhance_pretrain.pt'
    else:
        load_opt = True

    # 初始化主网络（生成器）
    net, optim_gen, weight_scheduler = model_init(step=step, load_path=load_path, load_opt=load_opt)

    # 初始化判别器及优化器（新增）
    disc = Discriminator().to(device)
    optim_disc = torch.optim.Adam(disc.parameters(), lr=c.gan_lr, betas=c.betas)

    # 加载已有GAN参数（如果有）
    if load_path != '' and os.path.exists(load_path):
        state_dicts = torch.load(load_path)
        if 'disc' in state_dicts:
            disc.load_state_dict(state_dicts['disc'])
        if 'opt_disc' in state_dicts and load_opt:
            optim_disc.load_state_dict(state_dicts['opt_disc'])
        print(f'Loaded GAN parameters from {load_path}')

    # 启动训练（传入GAN组件）
    visualizer = viz.Visualizer(c.loss_names)
    train(net, disc, optim_gen, optim_disc, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer,
          expinfo=expinfo, lam=lam)

    # 测试（不涉及GAN训练）
    train_epoch(net, disc, optim_gen, optim_disc, step, attack_method=attack_method, mode='test')
    calculate_PSNR_SSIM.main(f'{expinfo}')


if __name__ == '__main__':
    attack_method = 'mix2'
    lambda_c = 1.0
    lambda_s = 1.0
    lam = (lambda_c, lambda_s)
    for step in range(1,3):
        main(attack_method, step, start_epoch=0, end_epoch=1600, lam=lam)




