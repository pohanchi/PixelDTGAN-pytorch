from __future__ import print_function
import argparse
import os
import random
import pickle
import time
import cv2 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model_1 import *

from PIL import Image
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


def rescale_normalize_totensor(image_path):
    input_ = Image.open(image_path)
    image=np.array(input_)
    h, w = image.shape[:2]
    output_size = 64

    if h > w:
        new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size * h / w, output_size      
    new_h, new_w = int(new_h), int(new_w)

    img = transform.resize(image, (new_h, new_w),mode='constant')
    ishow = np.zeros((new_h,new_w,3)) 

    ishow[:,:,0] = img[:,:,2]*255                                                 
    ishow[:,:,1] = img[:,:,1]*255
    ishow[:,:,2] = img[:,:,0]*255

    cv2.imwrite('results/'+image_path[:5]+'_input_'+image_path[5:],ishow)
    img= (img*2)-1
    image_totensor = np.zeros((3,64,64))

    image_totensor[:] = 1.0
    img=img.transpose(2,0,1)
    if img.shape[1]<64:
        temp = 64 - img.shape[1]
        front = int((64 - (img.shape[1]))/2)
        end = temp - front
        image_totensor[:,front:-end,:img.shape[2]] = img

    elif img.shape[2] < 64:
        temp = 64 - img.shape[2]
        front = int((64-(img.shape[2]))/2)
        end = temp - front
        image_totensor[:,:img.shape[1],front:-end] = img

    image_totensor=image_totensor[np.newaxis,:,:,:]
    img_ = torch.from_numpy(image_totensor)
    return img_

def rescale(image_path):
    input_ = Image.open('Wanted_2/'+i)
    image=np.array(input_)
    h, w = image.shape[:2]
    output_size = 64
    if h > w:
        new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size * h / w, output_size      
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(image, (new_h, new_w),mode='constant')
    return img

def swith_channel(img):
    new_h,new_w = img.shape[:2]
    ishow = np.zeros((new_h,new_w,3)) 
    ishow[:,:,0] = img[:,:,2]*255                                                 
    ishow[:,:,1] = img[:,:,1]*255
    ishow[:,:,2] = img[:,:,0]*255
    return ishow

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=64, help='loadSize')
    parser.add_argument('--name', type=str, default='', help='test.jpg')
    parser.add_argument('--fineSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')   # 25 -> 1 
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    opt = parser.parse_args()
    print(opt)
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    netG = Generator(opt).to(device)

    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG,map_location='cpu'))
        netG.to(device)
    print(netG)

    if os.path.isdir('results'):
        print('results dir is existed')
    else:
        os.makedirs('results')
    netG_test=netG.eval()
    # for i in tqdm.tqdm(os.listdir('Wanted_2')):
    #     if i[10:16] == 'CLEAN0':
    #         img_=rescale_normalize_totensor('Wanted_2/'+i)
    #         result = netG_test.forward(img_.float())
    #         result = result.detach().numpy()
    #         result=(result[0,:,:,:] +1 )/2
    #         final=result.transpose(1,2,0)
    #         out = swith_channel(final)
    #         cv2.imwrite('results/'+i[0:26]+'_test_'+i[26:],out)
    #     elif i[10:16] == 'CLEAN1':
    #         img = rescale('Wanted_2/'+i)
    #         ishow =swith_channel(img)
    #         cv2.imwrite('results/'+i[0:26]+'_label_'+i[26:],ishow)
    img_=rescale_normalize_totensor(opt.name)
    result = netG_test.forward(img_.float())
    result = result.detach().numpy()
    result=(result[0,:,:,:] +1 )/2
    final=result.transpose(1,2,0)
    out = swith_channel(final)
    cv2.imwrite('results/'+opt.name[0:5]+'_test_'+opt.name[5:],out)




