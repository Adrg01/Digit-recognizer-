import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax
import numpy as np

# from constant import *

# import torch.nn.functional as F
# from constant import *

# model = torchvision.models.alexnet(pretrained=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class model(nn.Module):
    """
    Network designed for 128*128 input image
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(784, 400)
        self.linear2 = nn.Linear(196, 196)
        self.linear3 = nn.Linear(196, 50)
        self.linear4 = nn.Linear(50, 10)
        self.architecture = nn.Sequential(
            self.conv1,
            self.max,
            self.relu,
            self.flatten,
            # self.linear1,
            # self.relu,
            self.linear2,
            self.relu,
            self.linear3,
            self.relu,
            self.linear4,
            self.relu,
            nn.Softmax(),
        )

    def forward(self, x):
        logits = self.architecture(x)
        self.output = np.argmax(logits.detach().numpy())
        return logits


class model_easy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(784, 10)
        self.relu = nn.ReLU()
        self.architecture = nn.Sequential(self.linear, self.relu, nn.Softmax(dim=0))

    def forward(self, x):
        logits = self.architecture(x)
        self.output = np.argmax(logits.detach().numpy())
        return logits

    def eval(self, image):
        self(image)
        return self.output


def train_iter(model, image, label, loss_fn, optimizer):
    optimizer.zero_grad()
    pred = model(image)
    loss = loss_fn(pred, label)

    loss.backward()
    optimizer.step()

    return loss


# def train_batch(model,train_data,labels,loss_fun,optimizer,batch_size):
#     for i,(image,label) in enumerate(zip(train_data,labels)):
#         image =
