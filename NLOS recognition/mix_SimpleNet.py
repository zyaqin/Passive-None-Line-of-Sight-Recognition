import load1
import torch
import os
from models import SimpleNet
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
transform = transforms.ToTensor()
file_root=r'..\data\Mixed_setup_Measurement'
shuju_name=sorted(os.listdir(file_root), key=lambda x: float(x.split('_')[-1]))
save_path = file_root.split('\\')[-1]+'1'
if not os.path.exists(save_path):
    os.makedirs(save_path)
def random_split(dataset,ratio=0.8):
    train = []
    test = []
    for i in range(10):
        np.random.seed(42)
        data = torch.utils.data.Subset(dataset, (dataset.test_labels == i).nonzero().t()[0].tolist())
        train_size = int(ratio * len(data))
        test_size = len(data) - train_size
        train_db, val_db = torch.utils.data.random_split(data, [train_size, test_size])
        train.append(train_db)
        test.append(val_db)
    train_data= torch.utils.data.ConcatDataset(train)
    test_data=torch.utils.data.ConcatDataset(test)
    print(len(train_data),len(test_data))
    return train_data,test_data
def shuju_split_contact(name):
    train = []
    test=[]
    for i in range(len(name)):
        test_dataset = load1.MNIST(root=file_root, \
                                   test_file=os.path.join(save_path, name[i] + r'_test.pt'), \
                                   filename=name[i],
                                   train=False, transform=transform, download=True)
        torch.manual_seed(42)
        train_data,test_data=random_split(test_dataset)
        train.append(train_data)
        test.append(test_data)
    #print(len(train),len(test))
    return train, test
train, test=shuju_split_contact(shuju_name)
train_data= torch.utils.data.ConcatDataset(train)
test_data= torch.utils.data.ConcatDataset(test)
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 30
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=True)
#torch.cuda.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=SimpleNet()
model.to(device)

#实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.9
        print(optimizer.param_groups[0]['lr'])
    #for batch, (x, x_ori, y) in enumerate(test_loader):
    for img,label in train_loader:
        img = img.to(device)
        label = label.to(device)
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # net.eval() # 将模型改为预测模式
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                  eval_loss / len(test_loader), eval_acc / len(test_loader)))
# 将model保存为graph
torch.save(model.state_dict(), '../models/mixed_all_canshu_simplenet.pkl')
torch.save(model, '../models/mixed_all_model_simplenet.pkl')  # 保存整个神经网络的结构和模型参数
def draw_train_val(epochs, train_loss, val_loss, train_acc, val_acc):
    plt.subplot(2, 1, 1)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(epochs, train_loss, color='red', label='training loss')
    plt.plot(epochs, val_loss, color='blue', label='validation loss')
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(epochs, train_acc, color='red', label='training acc')
    plt.plot(epochs, val_acc, color='blue', label='validation acc')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
draw_train_val(np.arange(num_epoches),losses,eval_losses,acces,eval_acces)

