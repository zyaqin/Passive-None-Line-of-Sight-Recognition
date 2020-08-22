import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import scipy.io as sio
import torch
disturb=torch.load(r'D:\Passive None-Line-of-Sight Recognition\Attacks\disturb1.pt')#disturb1表示60000张训练集上模拟图与实验图差距的均值。具体计算方法请观看a.py文件。
data = sio.loadmat(r'D:\Passive None-Line-of-Sight Recognition\simA.mat')   # 加载mat文件
simA=torch.Tensor(data['simA'])
def fake_chenge(x):
    #return torch.mm(simA, x.transpose(1, 2).reshape(total, 1)).reshape(1, 128, 128).transpose(2, 1) / 25500
    return torch.mm(simA, x.transpose(1, 2).reshape(total, 1)).reshape(1, 128, 128).transpose(2, 1)/25500+disturb
change=transforms.Lambda(fake_chenge)
num_blocks=[32,40]
total=num_blocks[0]*num_blocks[1]
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 16, 3, 1, 1),  # padding=2保证输入输出尺寸相同
            nn.BatchNorm2d(16),
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(16, 16, 3, 1, 1),  # padding=2保证输入输出尺寸相同
            nn.BatchNorm2d(16),
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv3 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(16, 32, 3, 1, 1),  # padding=2保证输入输出尺寸相同
            nn.BatchNorm2d(32),
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv4 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(32, 32, 3, 1, 1),  # padding=2保证输入输出尺寸相同
            nn.BatchNorm2d(32),
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.fc1 = nn.Sequential(
            #nn.Linear(128, 64),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            #nn.Linear(64, 32),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Linear(512, 10)
        #self.fc3 = nn.Linear(32, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x= torch.stack([change(i.cpu()).cuda() for i in x], 0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # F.softmax(x, dim=1)