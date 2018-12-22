import numpy as np
from PIL import Image
import os, sys
import torch
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import pickle
import math

# Transformn the data
Landmarks = (pd.read_csv("./Data/train.csv")).values
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

# Aug data
Aug = (pd.read_csv("./Data/Count.csv")).values
Names = [i for i in Aug[:,0]]
Occurences = [i for i in Aug[:,1]]

#Transform the data
CenterCrop = transform.CenterCrop((170,170))
ColorJitter = transform.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.2)
RandomCrop = transform.RandomCrop((190,190), padding=0, pad_if_needed=True)
RandomRotation = transform.RandomRotation(5, resample=False, expand=False, center=None)
Hflip = transform.RandomHorizontalFlip(p=1)
Same = transform.CenterCrop((224,224))

Norm = transform.Compose([transform.Resize((224,224)),transform.ToTensor(), transform.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
Gray = transform.Grayscale(num_output_channels=3)

To_tensor = transform.ToTensor()
Resize = transform.Resize((224,224))

List = [CenterCrop, ColorJitter, RandomCrop, RandomRotation, Hflip, Same]

#Importing a pretrained model
resnet34 = models.resnet34(pretrained=True)
resnet34.fc = nn.Linear(512,5005)
#Importing the key/answer whales
Pic = []
Pic_Answer = []
#Seting up the training
batch = 0
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(resnet34.parameters(), lr=learning_rate, weight_decay=0.1)
# Set model to training mode
resnet34 = resnet34.train()
#Iterating
#Ploting the loss
plt.ion()
plt.xlabel("Iterations")
plt.ylabel("Loss")
axis = plt.gca()
axis.set_ylim([0,13.5])
plt.show()
path = "./Data/Images/Train"
dirs = os.listdir( path )
for j, i in enumerate(dirs):
    #Finding the index of an image
    index = Pictures.index(str(i))
    #Finding the tag of an image
    Tag = Tags[index]
    Answer = dict[str(Tag)]
    if Answer == 3:
        continue

    #Open the image
    img = Image.open("./Data/Images/Train/" + str(i))
    #Adding layer to one layer image
    img = Gray(img)
    #Resize the image
    img_a = Norm(img)
    #Appending the image and the answer to the lists
    Pic.append(img_a)
    Pic_Answer.append(Answer)


    if len(Pic) >= 64:
        Pic = torch.stack(Pic)
        Pic_Answer = torch.LongTensor(Pic_Answer)
        optimizer.zero_grad()
        outputs = resnet34(Pic)
        loss = criterion(outputs, Pic_Answer)
        print("Iters: {} || Loss: {} || Output: {}".format(j,loss.item(),outputs.shape))
        #Ploting the loss
        plt.plot(batch,loss.item(), 'X')
        plt.draw()
        plt.pause(0.0001)
        loss.backward()
        optimizer.step()
        batch += 1
        Pic = []
        Pic_Answer = []
    if j % 500 == 0:
        torch.save(resnet34.state_dict(), 'resnet34.pkl')

torch.save(resnet34.state_dict(), 'resnet34.pkl')
