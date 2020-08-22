'''
寻找扰动的数据加载器
'''
from __future__ import print_function
import cv2
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
class MNIST(data.Dataset):
    def __init__(self, root,test_file,filename,train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.test_file=test_file
        self.filename=filename
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.test_data, self.test_labels = torch.load(self.test_file)

    def __getitem__(self, index):
        #img_ori,img, target = self.test_data_ori[index],self.test_data[index], self.test_labels[index]
        img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        #img_ori = Image.fromarray(img_ori.numpy(), mode='L')


        if self.transform is not None:
            img = self.transform(img)
            #img_ori=self.transform(img_ori)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.test_file)

    def download(self):
        if self._check_exists():
            return
        # process and save as torch files
        #print('Processing...')
        test_set = (#read_image_file(r'D:\Passive None-Line-of-Sight Recognition\data\test'),
            read_image_file(os.path.join(self.root,self.filename)),
            read_label_file(r'D:\Passive None-Line-of-Sight Recognition\data\raw\t10k-labels-idx1-ubyte'))
        with open(self.test_file, 'wb') as f:
            torch.save(test_set, f)

        #print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)
def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()
def read_image_file(path):
    img_names = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))
    #img_names = sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))
    pic = []
    for i in range(len(img_names)):
        img = cv2.imread(os.path.join(path, img_names[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pic.append(torch.from_numpy(img))
    return torch.stack(pic, 0)



