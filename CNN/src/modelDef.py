import torch.nn as nn
import torch.nn.functional as F

# image input size: batch_size x 1 x 28 x 28
# batch_size: [int] 512
# dim 1: 1 channel, as images are between black and white
# dim 28: image pixel, also input units


class CNN_model(nn.Module):

    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)    # input channel, output channel, kernel size
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d((2, 2))    # kernel size: 2 x 2, stride: 2
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        size = x.size(0)
        output = self.conv1(x)              # output size: 24 x 24, now size: 512 x 10 x 24 x 24
        output = self.relu(output)
        output = F.max_pool2d(output, 2, 2)          # output size: 12 x 12, now size: 512 x 10 x 12 x 12
        output = self.conv2(output)         # output size: 10 x 10, now size: 512 x 20 x 10 x 10
        output = self.relu(output)
        output = output.view(size, -1)      # now size: 512 x 2000
        output = self.fc1(output)           # now size: 512 x 500
        output = self.relu(output)
        output = self.fc2(output)           # now size: 512 x 10
        out = self.softmax(output)
        return out
