import torch.nn as nn
import torch
from Helper.Data import *
from Helper.Models import resnet101
import matplotlib.pyplot as plt
import time
#Set the random seed
torch.cuda.manual_seed(seed)
#pyplot
plt.ion()
plt.xlabel("Iterations")
plt.ylabel("Loss")
axis = plt.gca()
axis.set_ylim([0,1.5])
plt.show()
for i in range(20):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(resnet101.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.09)

    #Iteratin
    iters = 0
    loss_cumul = 0
    for sample in dataloader:
        iters += 1
        x, y = sample["image"], sample["landmarks"]
        y = y.to(torch.int64)
        y = y.view(32)
        #Clear the gradient from the previous iteration
        optimizer.zero_grad()
        #Feed Forward
        outputs = resnet101(x)
        #Calculate the Loss
        loss = criterion(outputs,y)
        loss_cumul += loss
        #Getting the derivation of the gradients
        loss.backward()
        #Update the parameters
        optimizer.step()
        #Calculating the accuracy
        if iters % 10 == 0:
            # print("Iteration: {} || Loss: {}".format(iters,loss.item()))
            loss_cumul /= 10
            print("Alpha: {} || Loss: {}".format(learning_rate,loss_cumul))
            loss_cumul = 0
            learning_rate *= 10
            plt.plot(learning_rate,loss_cumul, 'X')
            plt.draw()
            plt.pause(0.0001)

            break
        #if iters % 50 == 0:
            # plt.plot(iters,loss.item(), 'X')
            # plt.draw()
            # plt.pause(0.0001)
            #torch.save(resnet101.state_dict(), './Saved_models/resnet101_2.pkl')
