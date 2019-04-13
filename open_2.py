import pickle 
import pickle 
import os 
import cv2
from shutil import copyfile
import argparse
import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=0, help='which index kind you want to start! You can continue selecting data kind with this')
parser.add_argument('--d_path',type=str, default='Wanted/', help='origin_data_dir')
opt = parser.parse_args()

p=pickle.load(open('stocastic_select.p','rb'))
print(len(p))
mypath = opt.d_path
want = 'Wanted_2/'
nowant = 'NoWanted_2/'
if os.path.isdir(want):
    print('Wanted dir is existed')
else:
    os.makedirs(want)
if os.path.isdir(nowant):
    print('NoWanted dir is existed')
else:
    os.makedirs(nowant)

for i in tqdm.tqdm(range(len(p))):
    if i < opt.index:
        continue
    else:
        target_name  = mypath + (p[i])['label']
        img = cv2.imread(target_name)
        cv2.imshow('image',img)
        k=cv2.waitKey(0)
        if k & 0xFF == ord('y'):
            copyfile(target_name,want+(p[i])['label'])
            cv2.destroyAllWindows()
            for j in (p[i])['data']:
                model_name = mypath + j
                img_ = cv2.imread(model_name)
                cv2.imshow('image',img_)
                tmp = cv2.waitKey(0)
                if tmp & 0xFF == ord('y') :
                    copyfile(model_name,want+j)
                    cv2.destroyAllWindows()
                if tmp & 0xFF == ord('n'):
                    copyfile(model_name,nowant+j)
                    cv2.destroyAllWindows()
                if tmp & 0xFF == ord('e') :
                    print('index= {} is finished, You must have {} kinds of data to label'.format(i,(len(p)-i)))
                    break;
        if k & 0xFF  == ord('n'):
            copyfile(target_name,nowant+(p[i])['label'])
            cv2.destroyAllWindows()
            for t in (p[i])['data']:
                model_name = mypath + t
                copyfile(model_name,nowant+t)
        if k & 0xFF  == ord('e'):
            print('index= {} is finished, You must have {} kinds of data to label'.format(i,(len(p)-i)))
            break
