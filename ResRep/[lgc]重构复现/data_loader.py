import torchvision.transforms as transforms
import torch
import torchvision

allowDL = False


def get_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainLoader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./datasets', train=True,
            download=allowDL, transform=transform
        ),
        batch_size=128,
        shuffle=True
    )

    testLoader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./datasets',
            train=False,
            download=allowDL,
            transform=transform
        ),
        batch_size=128,
        shuffle=False
    )

    nameClasses = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainLoader, testLoader, nameClasses

if __name__ == "__main__":
    get_loader()