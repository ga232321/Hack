import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import numpy as np
import glob
import cv2
import imageio
import pandas as pd
import os
import mopddd
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings

from torch.optim import Adam
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

class NatureDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pic = self.df.iloc[index]["pic"]
        ans = self.df.iloc[index]["answers"]
        # print(pic)
        pic = np.array(Image.open(pic).convert("RGB"))
        if self.transform is not None:
            augmentations = self.transform(image = pic)
            pic = augmentations

        return pic["image"], ans



df = pd.DataFrame()
# klikun - 0, малой -1 , шипун - 2

df["pic"] = os.listdir(r"C:\Users\ga232\Природа\root\klikun\images")[:3000] + os.listdir(r"C:\Users\ga232\Природа\root\Maliy\images")[:3000] + os.listdir(r"C:\Users\ga232\Природа\root\Shipun\images")[:3000]
answers = np.append(np.array([0 for i in range(3000)]),np.array([1 for i in range(3000)]))
answers = np.append(answers,np.array([2 for i in range(3000)]))
df["answers"] = answers


df["pic"].iloc[:3000] = df["pic"].iloc[:3000].apply(lambda x: f"C://Users//ga232//Природа//root//klikun//images//{x}" )
df["pic"].iloc[3000:6000] = df["pic"].iloc[3000:6000].apply(lambda x: f"C://Users//ga232//Природа//root//Maliy//images//{x}" )
df["pic"].iloc[6000:] = df["pic"].iloc[6000:].apply(lambda x: f"C://Users//ga232//Природа//root//Shipun//images//{x}" )


train, test = train_test_split(df, test_size=0.2)

train_transform = A.Compose(
    [

        A.Resize(380, 380),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(380, 380),
        ToTensorV2(),

    ]
)




train_dataset = NatureDataset(train, transform=train_transform)
test_dataset = NatureDataset(test, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, pin_memory=True,num_workers = 12,persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=True, pin_memory=True,num_workers = 4,persistent_workers=True )






model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





model.classifier.fc = nn.Linear(in_features=1792, out_features=3, bias=True)
model.requires_grad_(False)
model.classifier.requires_grad_(True)
model.features.requires_grad_(True)


model = model.to(device)

optimizer = Adam(
    [
        {'params': model.features.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ]
)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0,4.0]).to(device))
scheduler = StepLR(optimizer, step_size=5, gamma=0.2)


@torch.inference_mode()
def evaluate(model, loader) -> tuple[float, float]:
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.float().to(device), y.type(torch.LongTensor).to(device)

        output = model(x)

        loss = loss_fn(output, y)

        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy




def plot_stats(
        train_loss: list[float],
        valid_loss: list[float],
        train_accuracy: list[float],
        valid_accuracy: list[float],
        title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + ' accuracy')

    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(valid_accuracy, label='Valid accuracy')
    plt.legend()
    plt.grid()

    plt.show()




def whole_train_valid_cycle(model, num_epochs, title):
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model)
        valid_loss, valid_accuracy = evaluate(model, test_loader)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

        clear_output()

    plot_stats(
        train_loss_history, valid_loss_history,
        train_accuracy_history, valid_accuracy_history,
        title
    )



from tqdm import tqdm


def train(model) -> float:
    model.train()

    train_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.float().to(device), y.type(torch.LongTensor).to(device)
        #print(x,y)
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    train_loss /= len(train_loader)
    accuracy = correct / total

    return train_loss, accuracy








def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':

    whole_train_valid_cycle(model, 50, 'tunnedResNet')
    run()
    torch.save(model.state_dict(), "efficientNETModel.pt")