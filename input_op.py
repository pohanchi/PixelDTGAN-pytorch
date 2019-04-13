import os
import numpy as np
import random
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torchvision
from tensorboardX import SummaryWriter
import pickle
from skimage import io, transform


class mydataset(Dataset):
  def __init__(self,filepath,opt,transform):
    self.loadSize = opt.loadSize
    self.fineSize = opt.fineSize
    self.root = filepath
    self.transform = transform
    data=pickle.load(open(filepath,'rb'))
    print('data_length=',len(data))
    self.x = data[:18816]
    random.shuffle(self.x)
    self.X_train = np.asarray([i['data'] for i in self.x])
    self.y_train = np.asarray([i['label'] for i in self.x])  
    self.fake_train = np.asarray([i['not_label'] for i in self.x]) 
    return
  def __getitem__(self, index):
    #image_path='dataset/lookbook/data/'  #dataset file path
    image_path = 'wanted/'
    label = self.y_train[index]

    label_name = image_path+label

    image_name=image_path + self.X_train[index]

    not_label_name = image_path + self.fake_train[index]

    img = io.imread(image_name)
    img_label = io.imread(label_name)
    not_label = io.imread(not_label_name)
    
    associ_img = np.asarray(img_label)
    in_image   = np.asarray(img) 
    noassoci_img= np.asarray(not_label)
    data_type ={'input': in_image, 'associ': associ_img, 'noassoci': noassoci_img}
    
    if self.transform:
        data_type=self.transform(data_type)
    
    return  data_type
      
  def __len__(self):
    return len(self.x)

class Rescale(object):

  """Rescale the image in a sample to a given size.

  Args:
      output_size (tuple or int): Desired output size. If tuple, output is
          matched to output_size. If int, smaller of image edges is matched
          to output_size keeping aspect ratio the same.
  """
  def __init__(self, output_size):
      assert isinstance(output_size, (int, tuple))
      self.output_size = output_size

  def __call__(self, sample):

    image, label, no_label = sample['input'], sample['associ'],sample['noassoci']

    h, w = image.shape[:2]
    h_1, w_1 = label.shape[:2]
    h_2, w_2 = no_label.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
          new_h, new_w = self.output_size, self.output_size * w / h
      else:
          new_h, new_w = self.output_size * h / w, self.output_size
      if h_1 > w_1:
          new_h_1, new_w_1 = self.output_size, self.output_size * w_1 / h_1
      else:
          new_h_1, new_w_1 = self.output_size * h_1 / w_1, self.output_size
      if h_2 > w_2:
          new_h_2, new_w_2 = self.output_size, self.output_size * w_2 / h_2
      else:
          new_h_2, new_w_2 = self.output_size * h_2 / w_2, self.output_size
    else:
      new_h, new_w = self.output_size
      new_h_1, new_w_1 = self.output_size
      new_h_2, new_w_2 = self.output_size
    
    new_h, new_w = int(new_h), int(new_w)
    new_h_1,new_w_1 = int(new_h_1), int(new_w_1)
    new_h_2,new_w_2 = int(new_h_2), int(new_w_2)
    
    img = transform.resize(image, (new_h, new_w),mode='constant',anti_aliasing=False)
    
    label = transform.resize(label, (new_h_1,new_w_1),mode='constant',anti_aliasing=False)
    
    no_label=transform.resize(no_label,(new_h_2,new_w_2),mode='constant',anti_aliasing=False,)

    #print('image',image[:2,:2,:])
    #print('img',img[:2,:2,:])
    #print('img_end',((img*2)-1)[:2,:2,:])
    #print('------------------------')    
    return {'input': (img*2)-1, 'associ': (label*2)-1, 'noassoci':(no_label)*2-1}

class ToTensor(object):
  """Convert ndarrays in samplcriterion = nn.BCELoss() to Tensors."""

  def __call__(self, sample):

    image, label, no_label = sample['input'], sample['associ'], sample['noassoci']

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    label = label.transpose((2, 0, 1))
    no_label = no_label.transpose((2, 0, 1))
    image_totensor = np.zeros((3,64,64))
    label_totensor = np.zeros((3,64,64))
    nolabel_totensor = np.zeros((3,64,64))
    # print('image',image[:,:3,:3])
    # print('label_shape',label.shape)
    # print('nolabel_shape',no_label.shape)
    image_totensor[:] = 1.0
    label_totensor[:] = 1.0 
    nolabel_totensor[:] = 1.0
    if image.shape[1]<64:
        temp = 64 - image.shape[1]
        front = int((64 - (image.shape[1]))/2)
        #print('front',front)
        end = temp - front
        #print('end',end)
        image_totensor[:,front:-end,:image.shape[2]] = image
    elif image.shape[2] < 64:
        temp = 64 - image.shape[2]
        front = int((64-(image.shape[2]))/2)
        #print('front',front)
        end = temp - front
        #print('end',end)
        image_totensor[:,:image.shape[1],front:-end] = image
    if  label.shape[1]<64:
        temp = 64 - label.shape[1]
        front = int((64-(label.shape[1]))/2)
        #print('front',front)
        end = temp - front
        #print('end',end)
        label_totensor[:,front:-end,:label.shape[2]] = label
    elif label.shape[2] < 64:
        temp = 64 - label.shape[2]
        front = int((64 - label.shape[2])/2)
        #print('front',front)
        end = temp -front
        #print('end',end)
        label_totensor[:,:label.shape[1],front:-end] = label
    if  no_label.shape[1]<64:
        temp = 64 - no_label.shape[1]
        front = int((64-no_label.shape[1])/2)
        #print('front',front)
        end = temp -front
        #print('end',end)
        nolabel_totensor[:,front:-end,:no_label.shape[2]]= no_label
    elif no_label.shape[2] < 64:
        temp = 64 - no_label.shape[2]
        front = int((64-no_label.shape[2])/2)
        #print('front',front)
        end   = temp - front 
        #print('end',end)
        nolabel_totensor[:,:no_label.shape[1],front:-end]= no_label
    if (image.shape[1] == 64) & (image.shape[2] == 64):
        image_totensor[:,:64,:64] = image
    if (label.shape[1] == 64)  & (label.shape[2] == 64):
        label_totensor[:,:64,:64] = label
    if (no_label.shape[1] == 64) & (no_label.shape[2] == 64):
        nolabel_totensor[:,:64,:64] = no_label
        
    # print('image_totensor',image_totensor[:,:3,:3])
    # print('image_totensor_margin',image_totensor[:,63,63])
    #nolabel_totensor[:,:no_label.shape[1],:no_label.shape[2]] = no_label
    #label_totensor[:,:label.shape[1],:label.shape[2]]  = label 
    #print('totensor',image[:,:2,:2])
    return {'input': torch.from_numpy(image_totensor),
            'associ': torch.from_numpy(label_totensor),
            'noassoci':torch.from_numpy(nolabel_totensor)}

if __name__ == '__main__':
  mypath = "dataset/lookbook/data/"

  files = listdir(mypath)
  print(len(files))

  # for i,f in enumerate(files):
  #   if i == 4:
  #     temp=f[:9]
  #     tmp=cv2.imread(mypath+f)
  #     cv2.imshow('image',tmp)
  #     k=cv2.waitKey(0)
  #     if k == ord('s'):
  #       cv2.destroyAllWindows()
  #   if i >= 5 and f[:9]==temp and i<=77546:
  #     print("file_name_",i,"=", f)
  #     tmp=cv2.imread(mypath+f)
  #     cv2.imshow('image',tmp)
  #     k=cv2.waitKey(0)
  #     if k == ord('s'):
  #       cv2.destroyAllWindows()

  category = list()
  category_all =list()
  for i,f in enumerate(files):
    if i == 0:
      temp =f[:9]
      category += [temp]
    if temp != f[:9]:
      temp = f[:9]
      category += [temp]
  category=[i for i in set(category)]

  data =list()
  stocastic=list()
  dict_={'data':None,'label':None,'not_label':None}
  label_distribution={'data_num':None,'label':None}

  for i in category:
    for j,f in enumerate(files):
      if f[:9] == i and f[10:16]=='CLEAN1':
        category_all += [f]
        del files[j]

  for i in category_all:
    k=0
    tmp=[]
    for j,f in enumerate(files):
      if f[:9] == i[0:9]:
        k=k+1
        dict_['data']=f
        dict_['label']=i
        t = [ l for l in category_all if l != i]
        no_associ=random.choice(t)
        dict_['not_label']=no_associ
        tmp+=[f]
        data += [dict_.copy()]
        del files[j]
      else:
        pass
    label_distribution['data_num']= k
    label_distribution['data']= tmp
    label_distribution['label']=i
    stocastic += [label_distribution.copy()]
  print(data[:20])
  print(stocastic[:20])
  pickle.dump(data,open('data.p','wb'))
  pickle.dump(stocastic,open('stocastic.p','wb'))
  tmp=pickle.load(open('stocastic.p','rb'))
  print(tmp[:3])





  
  
      



