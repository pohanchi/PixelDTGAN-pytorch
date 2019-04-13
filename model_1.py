import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        print('x_shape',x.view(self.shape).size())
        return x.view(self.shape)

class Show(nn.Module):
    def __init__(self, *args):
        super(Show, self).__init__()
        self.shape = args

    def forward(self, x):
        print('x_show_shape',x.size())
        return x


#torch.set_num_threads(2)
class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        self.ngpu = opt.ngpu
        nc = 3
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)
        self.main = nn.Sequential(
            # input is (nc) x 64 x64
            nn.Conv2d(nc, ngf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Show(1,2,3),

            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, ngf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Show(1,2,3),

            # state size. (ngf*2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Show(1,2,3),
            
            # state size. (ngf*2) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Show(1,2,3),

            nn.Conv2d(ngf * 8, 64, 4, 4, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Show(1,2,3),

            nn.ConvTranspose2d(64,ngf*8, 4, 4,0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Show(1,2,3),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8,ngf*4, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Show(1,2,3),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4,ngf*2, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Show(1,2,3),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf*2,ngf, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Show(1,2,3),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf,nc, 5, 2,2, 1, bias=False),
            nn.Tanh(),
            # Show(1,2,3),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input_):
        if input_.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input_.float())
        return output

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.ngpu = opt.ngpu
        nc = 3
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1,0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_):
        if input_.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input_.float().cuda())
            

        return output.view(-1, 1).squeeze(1)

class Domain_Discriminator(nn.Module):

    def __init__(self, opt):
        super(Domain_Discriminator, self).__init__()
        self.ngpu = opt.ngpu
        nc = 3
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc*2, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1,0, bias=False),
            nn.Sigmoid()
        )
    def forward(self,input_):
        if input_.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input_.float().cuda())
            

        return output.view(-1, 1).squeeze(1)



