import torch.nn as nn
import torch
from Helper.Data import *
from Helper.Models import resnet152
import matplotlib.pyplot as plt
import time
#Move the resnet to the gpu
resnet152 = resnet152.cuda()
#Freeze all the layers
for param in resnet152.parameters():
    param.requires_grad = False
#Changing the last layer
resnet152.fc = nn.Linear(2048,5004)
# #pyplot
plt.ion()
plt.xlabel("Iterations")
plt.ylabel("Loss")
axis = plt.gca()
axis.set_ylim([0,1.5])
plt.show()

#Setting the loss/optin functions
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(resnet152.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.09)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.1)
iters = 0
for epoch in range(3):
    for sample in dataloader:
        iters += 1
        x, y = sample["image"], sample["landmarks"]
        y = y.to(torch.int64)
        y = y.view(x.size()[0])
        #move it to gpu
        x = x.cuda()
        y = y.cuda()
        #Clear the gradient from the previous iteration
        optimizer.zero_grad()
        #Feed Forward
        outputs = resnet152(x)
        #Calculate the Loss
        loss = criterion(outputs,y)
        #Getting the derivation of the gradients
        loss.backward()
        #Update the parameters
        optimizer.step()
        scheduler.step()
        if iters % 1000 == 0:
            #Pyplot
            plt.plot(iters,loss.item(), 'X')
            plt.draw()
            plt.pause(0.0001)

            #Print and save
            lr_get = 0
            for param_group in optimizer.param_groups:
                lr_get = param_group['lr']
            print("Loss: {} || Learning rate: {} || Iteration: {}".format(loss, lr_get, iters))
            torch.save(resnet152.state_dict(), './Saved_models/resnet152.pkl')
torch.save(resnet152.state_dict(), './Saved_models/resnet152.pkl')
