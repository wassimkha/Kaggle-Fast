from __future__ import print_function, division
import numpy as np
from PIL import Image
import os, sys
import torch
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import pandas as pd
import math

# Transformn the data
Landmarks = (pd.read_csv("./Data/New_Aug.csv")).values
Pictures = [i for i in Landmarks[:,0]]
Tags = [i for i in Landmarks[:,1]]
dict = {}
count = 0
for i in Tags:
    if i in dict:
        continue
    else:
        dict[i] = count
        count += 1
print(dict)
foo = input()

#Importing a pretrained model

# Set model to training mode

# Importing the key/answer whales
Pic = []
Pic_Answer = []
#Seting up the training
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(resnet101.parameters(), lr=learning_rate, weight_decay=0.001)
print(len(Pictures))
# Iterating
# Ploting the loss
plt.ion()
plt.xlabel("Iterations")
plt.ylabel("Loss")
axis = plt.gca()
axis.set_ylim([0,10])
plt.show()
path = "./Data/Images/Augmentation"
dirs = os.listdir( path )
for j, i in enumerate(Pictures):
    #Finding the index of an image
    index = Pictures.index(str(i))
    #Finding the tag of an image
    Tag = Tags[index]
    Answer = dict[str(Tag)]
    #Open the image
    img = Image.open("./Data/Images/Augmentation/" + str(i))
    #Resize the image
    img_a = Norm(img)
    #Appending the image and the answer to the lists
    Pic.append(img_a)
    Pic_Answer.append(Answer)

    if len(Pic) >= 32:
        Pic = torch.stack(Pic)
        Pic_Answer = torch.LongTensor(Pic_Answer)
        optimizer.zero_grad()
        outputs = resnet101(Pic)
        loss = criterion(outputs, Pic_Answer)
        print("Iters: {} || Loss: {} || Output: {}".format(j,loss.item(),outputs.shape))
        #Ploting the loss
        plt.plot(j,loss.item(), 'X')
        plt.draw()
        plt.pause(0.0001)
        loss.backward()
        optimizer.step()
        Pic = []
        Pic_Answer = []
        torch.save(resnet101.state_dict(), 'resnet101.pkl')
