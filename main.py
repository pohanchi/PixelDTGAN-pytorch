from __future__ import print_function
import argparse
import os
import random
import pickle
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from input_op import *
from tensorboardX import SummaryWriter
from skimage import io, transform
from model import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser()
#parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--loadSize', type=int, default=64, help='loadSize')
parser.add_argument('--fineSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=96)
parser.add_argument('--ndf', type=int, default=96)
parser.add_argument('--niter', type=int, default=65, help='number of epochs to train for')   # 25 -> 1 
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netA', default='', help="path to netA (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--epoch',type=int,default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
print(opt)
#print('num',torch.get_num_threads())
#torch.set_num_threads(2)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

cudnn.benchmark = True
print(torch.cuda.is_available())
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
dataset = mydataset('data.p',opt,transform=transforms.Compose([Rescale((64)),ToTensor()]))
train_dataloader=DataLoader(dataset= dataset,batch_size=opt.batchSize,shuffle=True,num_workers=4)
device = torch.device("cuda:0" if opt.cuda else "cpu")
print('device',device)
writer = SummaryWriter(opt.outf)

netG = Generator(opt).to(device).cuda()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(opt).to(device).cuda()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netA = Domain_Discriminator(opt).to(device).cuda()
netA.apply(weights_init)
if opt.netA != '':
    netA.load_state_dict(torch.load(opt.netA))
print(netA)



label       = torch.zeros((128,), requires_grad=False).to(device)
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr/3, betas=(opt.beta1,0.998))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr/2,   betas=(opt.beta1,0.998))
optimizerA = optim.Adam(netA.parameters(), lr=opt.lr/3, betas=(opt.beta1,0.998))
#optimizerD = optim.SGD(netD.parameters(), lr=opt.lr/3, momentum=opt.beta1)
#optimizerG = optim.SGD(netG.parameters(), lr=opt.lr/2, momentum=opt.beta1)
#optimizerA = optim.SGD(netA.parameters(), lr=opt.lr/3, momentum=opt.beta1)

criterion = nn.BCELoss().cuda()

real_label =1
fake_label =0
print(len(train_dataloader))
for epoch in tqdm.tqdm(range(opt.epoch-1,opt.niter)):
    for i, data in enumerate(train_dataloader):
    
        input_img = data['input']
        ass_label = data['associ']
        noass_label = data['noassoci']
        #print('input_tensor',input_img[0][1,:5,:5])
        #--- train with real
        lossD = 0
        netD.zero_grad()
        label.fill_(real_label)
        output_D =netD.forward(ass_label)
        errD_real1 = criterion.forward(output_D,label)
        lossD += errD_real1.item()
        #errD_real1 = errD_real1 * 1/3
        errD_real1.backward()
        

        # -- train with real (not associated)
        label.fill_(real_label)
        output_N = netD.forward(noass_label)

        errD_real2 = criterion.forward(output_N, label)
        lossD += errD_real2.item()
        #errD_real2 = errD_real2 * 1/3
        errD_real2.backward()

        #-- train with fake
        fake = netG.forward(input_img).detach()
        label.fill_(fake_label)
        output_f = netD.forward(fake)
        errD_fake = criterion.forward(output_f, label)
        lossD += errD_fake.item()
        #errD_fake = errD_fake * 1/3
        errD_fake.backward() 
        errD = (errD_real1 + errD_real2 + errD_fake)/3
        optimizerD.step()

        ############################
        # -- (1) Update A network
        ###########################
        lossA = 0
        netA.zero_grad()
        assd = torch.cat((input_img.float().cuda(), ass_label.float().cuda()), 1)
        noassd = torch.cat((input_img.float().cuda(), noass_label.float().cuda()), 1)
        fake = netG.forward(input_img).detach()
        faked = torch.cat((input_img.float().cuda(), fake.float().cuda()), 1)

        #-- train with associated
        label.fill_(real_label)
        output_Aa=netA.forward(assd)
        errA_real1 = criterion.forward(output_Aa, label)
        lossA += errA_real1.item()
        #errA_real1 = errA_real1 * 1/3
        errA_real1.backward()

        #-- train with not associated
        label.fill_(fake_label)
        output_An = netA.forward(noassd)
        errA_real2 = criterion.forward(output_An, label)
        lossA += errA_real2.item()
        #errA_real2 = errA_real2 * 1/3
        errA_real2.backward()

        #-- train with fake
        label.fill_(fake_label)
        output_Af = netA.forward(faked)
        errA_fake = criterion.forward(output_Af, label)
        lossA += errA_fake.item()
        #errA_fake = errA_fake * 1/3
        errA_fake.backward()

        errA = (errA_real1 + errA_real2 + errA_fake)/3
        optimizerA.step()

        ############################
        # -- (2) Update G network
        ###########################
        lossG = 0 
        netG.zero_grad()
        fake= netG.forward(input_img)
        output=netD.forward(fake)
        label.fill_(real_label) #-- fake labels are real for generator cost

        errGD = criterion.forward(output,label)
        lossG += errGD.item()
        #errGD = errGD * 1/2
        errGD.backward(retain_graph=True)

        faked = torch.cat((input_img.float().cuda(), fake.float().cuda()), 1)
        output_A = netA.forward(faked)
        label.fill_(real_label)#-- fake labels are real for generator cost
        errGA = criterion.forward(output_A,label)
        lossG += errGA.item()
        #errGA = errGA * 1/2
        errGA.backward()
        errG = (errGA + errGD)/2
        optimizerG.step()
        # print('batch_step',i+len(train_dataloader)*epoch)
        if (i==0) & (epoch == 0):
            start_time=time.time()
        if (i == 536) & (epoch == 0):
            end_time = time.time()
            print('1_epoch takes {} seconds'.format(end_time-start_time))
        if (i+len(train_dataloader)*epoch) % 67==0:
            print('epoch={},batch={}, lossG={}, lossA={}, lossD={}'.format(epoch,i,lossG/2,lossA/3,lossD/3))
            writer.add_scalar('data_adam/lossA', errA, i+len(train_dataloader)*epoch)
            writer.add_scalar('data_adam/lossD', errD, i+len(train_dataloader)*epoch)
            writer.add_scalar('data_adam/lossG', errG, i+len(train_dataloader)*epoch)
        if (i+len(train_dataloader)*epoch) % 134==0 :
            fake = (fake +1)/2
            ass_label = (ass_label +1)/2
            writer.add_image('data/generator_adam_loss_{} input_picture'.format(i+len(train_dataloader)*epoch),input_img[:8],i+len(train_dataloader)*epoch)
            writer.add_image('data/generator_adam_loss_{} fake_picture'.format(i+len(train_dataloader)*epoch),fake[:8],i+len(train_dataloader)*epoch)
            writer.add_image('data/generator_adam_loss_{} ground_truth'.format(i+len(train_dataloader)*epoch),ass_label[:8],i+len(train_dataloader)*epoch)
    # do checkpointing
    if (epoch+1) % 1 ==0:
        torch.save(netG.state_dict(), '%s/adam_netG_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(netD.state_dict(), '%s/adam_netD_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(netA.state_dict(), '%s/adam_netA_epoch_%d.pth' % (opt.outf, epoch+1))
writer.close()
