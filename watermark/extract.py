#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import calculate_PSNR_SSIM
import numpy as np
from model import *
import config as c
from torch.utils.tensorboard import SummaryWriter
import my_datasets
import viz
import warnings
from util import attack, gauss_noise, mse_loss, computePSNR, dwt, iwt
import torchvision

# 网络参数数量

def load(net, optim, name, load_opt=True):
    state_dicts = torch.load(name)
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
    # attack_container = attack(container_img, attack_method)
    attack_container = container_img
    input_container = dwt(attack_container)

    return container_img, attack_container, output_z, output_container, input_container

def train_epoch(net, step, optim=None, attack_method=None, i_epoch=None, writer=None, mode='train', lam=(1.0, 1.0), device='cuda'):
    dataloader = my_datasets.testloader
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            input_container = data.to(device)
            input_container = dwt(input_container)
            output_z = gauss_noise(input_container.shape)

            #################
            #   backward:   #
            #################
            output_rev = torch.cat((input_container, output_z), 1)
            output_image = net(output_rev, rev=True)
            extracted = output_image.narrow(1, 4 * c.channels_in,
                                            output_image.shape[1] - 4 * c.channels_in)
            extracted = iwt(extracted)
            if step == 1 or step == 2:
                extracted = net.post_enhance(extracted)

                extracted = extracted.clip(0, 1)



            if mode == 'test':
                print(i_batch)
                torchvision.utils.save_image(extracted, '/home/nas928/huangshuxin/lab/DIV2K/ours/gaussian10/extract/' + '%.5d.png' % i_batch)


    before = ''






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
    net.cuda()
    init_model(net)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    if load_path != '':
        net, optim = load(net, optim, load_path, load_opt)
        print(f'load from {load_path}')
    return net, optim, weight_scheduler



def main(attack_method, step, load_path='', start_epoch=0, end_epoch = 1600, lam=(1.0, 1.0)):
    warnings.filterwarnings("ignore")
    if step == 0:
        expinfo = f'{attack_method}_hinet_pretrain'
    elif step == 1:
        expinfo = f'{attack_method}_enhance_pretrain'
    elif step == 2:
        expinfo = f'{attack_method}_enhance_finetune'


    load_opt = True


    net, optim, weight_scheduler = model_init(step=step, load_path=load_path, load_opt=load_opt)

    visualizer = viz.Visualizer(c.loss_names)
    # train(net, step, optim, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer, expinfo=expinfo, lam=lam)
    train_epoch(net, attack_method=attack_method, mode='test', step=step)
    # calculate_PSNR_SSIM.main(f'{expinfo}')



if __name__ == '__main__':
    attack_method = 'mix2'
    lambda_c = 1.0
    lambda_s = 2.0
    lam = (lambda_c, lambda_s)
    for step in range(2,3):
        main(attack_method, step, load_path='/home/nas928/huangshuxin/lab/PRIS/model2_1/mix2_enhance_finetune.pt', start_epoch=0, end_epoch=1600, lam=lam)




