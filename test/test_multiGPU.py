#!/usr/bin/python3
# coding: utf-8
import torch
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

device_ids = [0, 1]
BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
data_train = datasets.MNIST(root = "/home/roit/datasets/mnist",
                            transform=transform,
                            train = True,
                            download = False)
data_test = datasets.MNIST(root="/home/roit/datasets/mnist",
                           transform = transform,
                           train = False)

data_loader_train = DataLoader(dataset=data_train,
                                                # 这里注意batch size要对应放大倍数
                                                batch_size = BATCH_SIZE * len(device_ids),
                                                shuffle = True,
                                                 num_workers=2)

data_loader_test = DataLoader(dataset=data_test,
                                               batch_size = BATCH_SIZE * len(device_ids),
                                               shuffle = True,
                                                num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(stride=2, kernel_size=2),
    )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),                            
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
    )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
model = torch.nn.DataParallel(model, device_ids=device_ids) # 声明所有可用设备
model = model.cuda(device=device_ids[0])  # 模型放在主设备
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 50
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
    for data in tqdm(data_loader_train):
        X_train, y_train = data
        # 注意数据也是放在主设备
        X_train, y_train = X_train.cuda(device=device_ids[0]), y_train.cuda(device=device_ids[0])
        
        outputs = model(X_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = X_test.cuda(device=device_ids[0]), y_test.cuda(device=device_ids[0])
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                                                                                      100*running_correct/len(data_train),
                                                                                      100*testing_correct/len(data_test)))
torch.save(model.state_dict(), "model_parameter.pkl")