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


#Transform the data
Norm = transform.Compose([transform.Resize((224,224)),transform.ToTensor(), transform.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
Gray = transform.Grayscale(num_output_channels=3)
To_tensor = transform.ToTensor()
#Importing a pretrained model
resnet34 = models.resnet34(pretrained=False)
resnet34.fc = nn.Linear(512,2)
resnet34.load_state_dict(torch.load('resnet34.pkl'))
resnet34 = resnet34.eval()
#Initialisation of variables
Pic = []
Pic_Answer = []
path = "./Data/Images/Train"
dirs = os.listdir( path )
#Iterating
correct = 0
total = 0
for j, i in enumerate(dirs):
    #Open the image
    img = Image.open("./Data/Images/Train/" + str(i))
    #Adding layer to one layer image
    x = To_tensor(img)
    if x.size()[0] == 1:
        img = Gray(img)
    #Resize the image
    img = Norm(img)
    #Finding the index of an image
    index = Pictures.index(str(i))
    #Finding the tag of an image
    Tag = Tags[index]
    #Appending the image and the answer to the lists
    Answer = dict[str(Tag)]
    if Answer != 3:
        Answer = 0
    if Answer == 3:
        Answer = 1
    img = img.view(1,3,224,224)
    output = resnet34(img)
    _,prediction = torch.max(output.data, 1)
    if prediction == Answer:
        correct += 1
        total += 1
    else:
        total += 1
    if total % 500 == 0:
        accuracy = float(correct) / total
        print("Iteration: {} || Accuracy: {}%".format(total,accuracy ))
