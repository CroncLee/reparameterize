from mynet import ResRepNet
import torch
import torch.nn as nn
from data_loader import get_loader
import torch.optim as optim
from tqdm import tqdm

net = ResRepNet()
net.cuda()

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
        for m in net.modules():
            if hasattr(m, 'pwc'):
                lasso_grad = 0.001 * m.pwc.weight.data * ((m.pwc.weight.data ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5))
                if m.pwc.weight.grad is None:
                    m.pwc.weight.grad = lasso_grad
                else:
                    m.pwc.weight.grad.data.add_(lasso_grad)
        loss.backward()
        optimizer.step()
        batch_tail += len(labels)
        correct += torch.sum(preds == labels.data).to(torch.float32)
    train_acc = correct / batch_tail
    print("\r epoch :" + str(epoch + 1) + ' train' + ' acc: ', train_acc.data)
torch.save(net.state_dict(), "./mynet.pth")