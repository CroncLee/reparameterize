import torch.optim as optim
import torch.nn as nn
from mynet import DbbCNNnet
from tqdm import tqdm
import torch
from data_loader import get_loader

net = DbbCNNnet()
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
trainLoader, testLoader, nameClasses = get_loader()

for epoch in range(100):
    correct = 0
    batch_tail = 0
    for i, (images, labels) in enumerate(tqdm(trainLoader)):
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_tail += len(labels)
        correct += torch.sum(preds == labels.data).to(torch.float32)
    train_acc = correct / batch_tail
    print("\r epoch :" + str(epoch + 1) + ' train' + ' acc: ', train_acc.data)

torch.save(net.state_dict(), "./mynet.pth")
