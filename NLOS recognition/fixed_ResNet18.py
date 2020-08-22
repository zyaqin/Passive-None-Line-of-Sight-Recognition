import numpy as np
import matplotlib.pyplot as plt
import load
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,models
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 30
transform = transforms.ToTensor()
#下载MNIST数据集
train_dataset = load.MNIST(root='../data/',train=True,transform=transform,download=False)
test_dataset = load.MNIST(root='../data/',train=False,transform=transform ,download=False)
#将下载的MNIST数据导入到dataloader中
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=train_batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=test_batch_size,shuffle=True)
'''
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
    return train_data,test_data
train_data,test_data=random_split(test_dataset)
#将下载的MNIST数据导入到dataloader中
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=True)
'''
'''
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(next(examples)[1][0])
print(example_data[0][0].shape)
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
'''

model=models.resnet18(pretrained=True)
stride,kernel_size,padding,bias=model.conv1.stride,model.conv1.kernel_size,model.conv1.padding,model.conv1.bias
model.conv1=nn.Conv2d(1,64,kernel_size,stride,padding,bias=bias)
model.avgpool=nn.AvgPool2d(4,stride=1)
print(model.avgpool)
model.fc = nn.Linear(512,10)

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
    for img, label in train_loader:
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
torch.save(model.state_dict(), r'../models/sy_Resnet30.pkl')
torch.save(model, r'../models/sy_Resnet30.pkl')  # 保存整个神经网络的结构和模型参数
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
