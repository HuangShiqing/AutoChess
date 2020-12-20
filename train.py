import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenet import mobilenet_v2
import torch.optim as optim
import torch.nn as nn

import os
import cv2
import pandas as pd
from PIL import Image


class CactiDataset(Dataset):
    def __init__(self, csv, dir, transform=None):
        super().__init__()
        self.csv = csv.values
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        img_name, label = self.csv[index]
        img_path = os.path.join(self.dir, img_name)
        # image = cv2.imread(img_path)
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def test():
    Gb_data_dir = "/home/hsq/DeepLearning/volume/gstreamer/process/img/"
    csv = pd.read_csv(Gb_data_dir + "valid.csv")
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    testset = CactiDataset(csv, Gb_data_dir, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load("./trained_model/model_latest.pkl")
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (
        len(testset), 100 * correct / total))
    exit()


def test_one(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path)
    image = image.convert('RGB')
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    net = torch.load("./model_600.pkl")
    net.eval()

    outputs = net(image)
    predicted = nn.functional.softmax(outputs,1).max(1)
    print("output tensor: ",predicted)
    exit()



if __name__ == '__main__':
    # test_one("/workspace/volume/gstreamer/process/classification/2.jpg")
    # test()

    Gb_data_dir = "/home/hsq/DeepLearning/volume/gstreamer/process/img/"
    csv = pd.read_csv(Gb_data_dir + "train.csv")
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), transforms.ToTensor()])#transforms.RandomCrop((300,300))

    trainset = CactiDataset(csv, Gb_data_dir, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = resnet50()
    # net.fc = nn.Linear(2048, 62)
    net = mobilenet_v2()
    net.classifier[1] = nn.Linear(1280, 62)
    # print(net)
    # exit()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1000):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        if epoch % 100 == 0:
            torch.save(net, './trained_model/model_{}.pkl'.format(epoch))
    torch.save(net, './trained_model/model_latest.pkl')
    print('Finished Training')

    exit()
