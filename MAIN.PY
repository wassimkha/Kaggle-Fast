import torch.nn as nn
import torch
from Helper.Data import *
from Helper.Models import resnet101

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(resnet101.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)

#Iteratin
for sample in dataloader:
    x, y = sample["image"], sample["landmarks"]
    print(x.size())
    #Clear the gradient from the previous iteration
    optimizer.zero_grad()
    print('+1')
    #Feed Forward
    outputs = resnet101(x)
    print('+1')
    #Calculate the Loss
    loss = criterion(outputs,y)
    print('+1')
    #Getting the derivation of the gradients
    loss.backward()
    print('+1')
    #Update the parameters
    optimizer.step()
    break
