import torch
import torch.nn as nn
from model import LeNet
from data import data_train_loader as data_train_loader
from torch.utils.tensorboard import SummaryWriter

model = LeNet()
model.train()
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

train_loss = 0
correct = 0
total = 0


for batch_idx, (inputs, targets) in enumerate(data_train_loader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(data_train_loader),
          'Loss:%.3f | Acc:%.3f%%(%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

save_info = {
    "iter_num": batch_idx,
    "optimizer": optimizer.state_dict(),
    "model": model.state_dict(),
}

save_path = "./model.pth"

torch.save(save_info, save_path)
