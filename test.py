import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import pandas as pd
from PIL import Image
import os

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

def test(model_path, Gb_data_dir):
    csv = pd.read_csv(Gb_data_dir + "valid.csv")
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    testset = CactiDataset(csv, Gb_data_dir, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load(model_path)
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


def test_one(model_path, img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path)
    image = image.convert('RGB')
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    net = torch.load(model_path)
    net.eval()

    outputs = net(image)
    predicted = nn.functional.softmax(outputs,1).max(1)
    print("output tensor: ",predicted)
    exit()

if __name__ == '__main__':
    model_path = "./trained_weapon_model/model_600.pkl"
    Gb_data_dir = "/home/hsq/DeepLearning/volume/gstreamer/process/AutoChess/log/weapon/"
    test(model_path, Gb_data_dir)

    # model_path = "./trained_weapon_model/model_600.pkl"
    # img_path = "/workspace/volume/gstreamer/process/classification/2.jpg"
    # test_one("/workspace/volume/gstreamer/process/classification/2.jpg")