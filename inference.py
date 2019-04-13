from __future__ import print_function
import argparse
import os
import random
import pickle
import time
import glob
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import final

from PIL import Image
import numpy as np
import scipy
#import imageio
import cv2
from tensorboardX import SummaryWriter
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--loadSize', type=int, default=64, help='loadSize')
parser.add_argument('--name', type=str, default='./inputs/orange.jpg', help='test.jpg')
parser.add_argument('--fineSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=96)
parser.add_argument('--ndf', type=int, default=96)
parser.add_argument('--niter', type=int, default=65, help='number of epochs to train for')   # 25 -> 1 
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
print(opt)



cudnn.benchmark = True
print(torch.cuda.is_available())
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")
print('device',device)
netG = final.final_cpu
netG.load_state_dict(torch.load('./final.pth'))
netG.eval()
#print(netG)


for filename in glob.glob('./Wanted_2/*.jpg'): #assuming gif
  print(filename)
  input_ = Image.open(filename)
  print(np.shape(input_))
  img=np.array(input_.resize((64,64),Image.BILINEAR))
  print(np.shape(img))
  #print(img)
  img = (img/255)*2 -1
  #print(img)
  img=img.transpose(2,0,1)
  img=img[np.newaxis,:,:,:]
  #print(img.shape)
  img_ = torch.from_numpy(img)
  img_=img_.type('torch.FloatTensor')
  result=netG.forward(img_)
  #print(result)
  print(result.shape)
  result = result[0,:,:,:]
  print(result.shape)
  result_ = result.detach().numpy()
  #print(result_.shape)
  #print(result_)
  result=result_.transpose(1,2,0)
  #print(result.shape)
  #print(result)
  res2 = result+1
  #print(res2.shape)
  #print(res2)
  res3 = res2/2
  #print(res3.shape)
  #print(res3)
  res4 = res3*255
  #print(res4.shape)
  #print(res4)
  res5 = res4.astype(int)
  #print(res5.shape)
  #print(res5)
  
  out = np.zeros((64,64,3))
  out[:,:,0] = res5[:,:,2]
  out[:,:,1] = res5[:,:,1]
  out[:,:,2] = res5[:,:,0]
  #print(out)
  #img_name = os.path.basename(filename)
  img_name = os.path.splitext(os.path.basename(filename))[0]
  
  cv2.imwrite('./results/'+img_name+'_result.jpg', out)
