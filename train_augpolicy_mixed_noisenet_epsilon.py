#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import timeit
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils import data
from datasets.datasets import LIPDataValSet
import networks
import utils.schp as schp
from datasets.datasets_augpolicy_mixed import LIPDataSet
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler
import pdb
from utils.miou import compute_mean_ioU

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if index % 1000 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = model(image.cuda())
            if gpus > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                parsing = outputs[0][-1]
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]


    return parsing_preds, scales, centers

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0,1')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log1')
    parser.add_argument("--model-restore", type=str, default='./log1/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    parser.add_argument('--noise2net', default=True, action='store_true')
    parser.add_argument('--noisenet-max-eps', default=0.75, type=float)
    parser.add_argument('--noisenet-prob', default=0.5, type=float)
    parser.add_argument('--aug-prob', default=0.5, type=float)


    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model Initialization
    AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    model = DataParallelModel(AugmentCE2P)
    model.cuda()
    print(model)
    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))

    restore_from = args.model_restore
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        checkpoint = torch.load(restore_from)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']-1

    SCHP_AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    schp_model.cuda()
    print(schp_model)
    if os.path.exists(args.schp_restore):
        print('Resuming schp checkpoint from {}'.format(args.schp_restore))
        schp_checkpoint = torch.load(args.schp_restore)
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = schp_checkpoint['cycle_n']
        schp_model.load_state_dict(schp_model_state_dict)

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # Data Loader
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    train_dataset = LIPDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform, aug_prob=args.aug_prob)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size * len(gpus),
                                   num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))


    # Data loader
    lip_test_dataset = LIPDataValSet(args.data_dir, 'val', crop_size=input_size, transform=transform, flip=False)
    num_samples = len(lip_test_dataset)
    print('Totoal testing sample numbers: {}'.format(num_samples))
    testloader = data.DataLoader(lip_test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                 eta_min=args.learning_rate / 100, warmup_epoch=10,
                                 start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / 2,
                                 cyclical_epoch=args.cycle_epochs)


    if args.noise2net:
        noise2net_batch_size = int(args.batch_size * len(gpus) * args.noisenet_prob)
        noise2net = Res2Net(epsilon=[0.50,0.50,0.50,0.50], hidden_planes=16, batch_size=noise2net_batch_size).train().cuda()
        print('noiseed_batch_size:',noise2net_batch_size)
        #print(noise2net)



    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()
    print('start_epoch:', start_epoch)
    for epoch in range(start_epoch, args.epochs):
        
        lr_scheduler.step(epoch=epoch)
        lr = lr_scheduler.get_lr()[0]

        model.train()



        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader) * epoch

            images, labels, _ = batch
            labels = labels.cuda(non_blocking=True)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)

            #pdb.set_trace()
            if args.noise2net:
                batch_size = images.shape[0]
                
                #
                with torch.no_grad():
                    # Setup network
                    noise2net.reload_parameters()
                    eps_list = []
                    for i_eps in range(4):
                        eps_list.append(random.uniform(args.noisenet_max_eps / 2.0, args.noisenet_max_eps))

                    noise2net.set_epsilon(eps_list)
                    # Apply aug
                    if(batch_size >= noise2net_batch_size):
                        bx_auged = images[:noise2net_batch_size].reshape((1, noise2net_batch_size * 3, input_size[0], input_size[1]))
                        #print(batch_size, '----', bx_auged.shape,'----',images.shape)
                        bx_auged = noise2net(bx_auged.cuda())
                        bx_auged = bx_auged.reshape((noise2net_batch_size, 3, input_size[0], input_size[1]))
                        images[:noise2net_batch_size] = bx_auged
                    else:
                        #pdb.set_trace()
                        bx_auged = images.repeat(2,1,1,1)
                        bx_auged = bx_auged[:noise2net_batch_size].reshape((1, noise2net_batch_size * 3, input_size[0], input_size[1]))

                        #print(i_iter, '----', batch_size, '----', bx_auged.shape,'----',images.shape)
                        bx_auged = noise2net(bx_auged.cuda())
                        bx_auged = bx_auged.reshape((noise2net_batch_size, 3, input_size[0], input_size[1]))
                        images[:batch_size] = bx_auged[:batch_size]                       


                    # torchvision.utils.save_image(unnorm_fn(bx[:5].detach().clone()).clamp(0, 1), os.path.join(args.save, "noise2net_groupconv.png"))
                    # exit()

            #pdb.set_trace()        
            preds = model(images)
            #pdb.set_trace()
            # Online Self Correction Cycle with Label Refinement
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds = schp_model((images.cuda())) #Variable
                    soft_parsing = [] # torch.zeros()
                    soft_parsing = []
                    soft_edge = []
                    for soft_pred in soft_preds:
                        soft_parsing.append(soft_pred[0][-1].cpu())
                        soft_edge.append(soft_pred[1][-1].cpu())

                    #for tmp in soft_parsing:

                    soft_preds = torch.cat(soft_parsing, dim=0)
                    soft_preds = soft_preds.cuda()
                    soft_edges = torch.cat(soft_edge, dim=0)
                    soft_edges = soft_edges.cuda()
            else:
                soft_preds = None
                soft_edges = None      

            loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 10 == 0:
                print('noise= {} iter = {} of {} epoch = {} of {} completed, lr = {}, loss = {}'.format(noise2net_batch_size, i_iter, total_iters,  epoch, args.epochs, lr,
                                                                             loss.data.cpu().numpy()))


        if (epoch + 1) % (args.eval_epochs) == 0:
            schp.save_schp_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, False, args.log_dir, filename='checkpoint_{}.pth.tar'.format(epoch + 1))

            parsing_preds, scales, centers = valid(model, testloader, input_size,  num_samples, len(gpus))
            mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
            print(mIoU)  

        # Self Correction Cycle with Model Aggregation
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print('Self-correction cycle number {}'.format(cycle_n))
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)
            schp.save_schp_checkpoint({
                'state_dict': schp_model.state_dict(),
                'cycle_n': cycle_n,
            }, False, args.log_dir, filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                             (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


########################################################################################################
### Noise2Net
########################################################################################################

import sys
import os
import numpy as np
import os
import shutil
import tempfile
from PIL import Image
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as trn
from torchvision import datasets
import torchvision.transforms.functional as trnF 
from torch.nn.functional import gelu, conv2d
import torch.nn.functional as F
import random
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, hidden_planes=9, scale = 4, batch_size=5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality (multiply by batch_size)
            planes: output channel dimensionality (multiply by batch_size)
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes * batch_size
        self.conv1 = nn.Conv2d(inplanes * batch_size, width*scale, kernel_size=1, bias=False, groups=batch_size)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            K = random.choice([1, 3])
            D = random.choice([1, 2, 3])
            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True, groups=batch_size))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * batch_size, kernel_size=1, bias=False, groups=batch_size)
        self.bn3 = nn.BatchNorm2d(planes * batch_size)

        self.act = nn.ReLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.hidden_planes = hidden_planes
        self.batch_size = batch_size

    def forward(self, x):
        _, _, H, W = x.shape
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out) # [1, hidden_planes*batch_size*scale, H, W]
        
        # Hack to make different scales work with the hacky batches
        out = out.view(1, self.batch_size, self.scale, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        
        spx = torch.split(out, self.width, 1) # [ ... (1, hidden_planes*batch_size, H, W) ... ]
        
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]),1)
        
        # Undo hack to make different scales work with the hacky batches
        out = out.view(1, self.scale, self.batch_size, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class Res2Net(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=16, batch_size=5):
        super(Res2Net, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)

    def reload_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()
 
    def set_epsilon(self, new_eps):
        self.epsilon = new_eps

    def forward_original(self, x):                
        x = (self.block1(x) * self.epsilon[0]) + x
        x = (self.block2(x) * self.epsilon[1]) + x
        x = (self.block3(x) * self.epsilon[2]) + x
        x = (self.block4(x) * self.epsilon[3]) + x
        return x

    def forward(self, x):
        return self.forward_original(x)



if __name__ == '__main__':
    main()
