import torch
import torch.nn as nn
import torch.optim as optim
import os

from dataPrep import prepMNIST
from modelDef import CNN_model


def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    lss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(data)
        ls = loss(out, label).sum()
        ls.backward()
        optimizer.step()
        lss = ls.item()

    save_path = '../res'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_path = os.path.join(save_path, 'res_CNN.txt')
    with open(res_path, 'a') as f:
        f.write("Train epoch: {}, loss: {:.4f}\n".format(epoch+1, lss))
    print("Train epoch: {}, loss: {:.4f}".format(epoch+1, lss))


def test(model, device, test_loader, loss):
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)   # size: 512 x 10
            test_loss += loss(out, label).sum().item()
            pred = out.argmax(1, keepdim=True)  # 找到概率最大的下标
            correct += out.argmax(1, keepdim=True).eq(label.view_as(pred)).sum().item()

    save_path = '../res'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_path = os.path.join(save_path, 'res_CNN.txt')
    with open(res_path, 'a') as f:
        f.write("Test loss: {:.4f}, accuracy: {:.4f}\n".format(test_loss, correct/len(test_loader.dataset)))
    print("Test loss: {:.4f}, accuracy: {:.4f}".format(test_loss, correct / len(test_loader.dataset)))


if __name__=='__main__':
    batch_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = prepMNIST(batch_size)
    model = CNN_model().to(device)
    optimizer = optim.Adam(model.parameters())
    loss = nn.NLLLoss()
    epochs = 20
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, loss, epoch)
        test(model, device, test_loader, loss)
