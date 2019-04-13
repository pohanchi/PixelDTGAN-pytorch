import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

final_cpu = nn.Sequential( # Sequential,
	nn.Conv2d(3,96,(4, 4),(2, 2),(1, 1),1,1,bias=False),
	nn.LeakyReLU(0.2),
	nn.Conv2d(96,192,(4, 4),(2, 2),(1, 1),1,1,bias=False),
	nn.BatchNorm2d(192),
	nn.LeakyReLU(0.2),
	nn.Conv2d(192,384,(4, 4),(2, 2),(1, 1),1,1,bias=False),
	nn.BatchNorm2d(384),
	nn.LeakyReLU(0.2),
	nn.Conv2d(384,768,(4, 4),(2, 2),(1, 1),1,1,bias=False),
	nn.BatchNorm2d(768),
	nn.LeakyReLU(0.2),
	nn.ConvTranspose2d(768,384,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(384),
	nn.ReLU(),
	nn.ConvTranspose2d(384,192,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(192),
	nn.ReLU(),
	nn.ConvTranspose2d(192,96,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(96),
	nn.ReLU(),
	nn.ConvTranspose2d(96,3,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.Tanh(),
)
