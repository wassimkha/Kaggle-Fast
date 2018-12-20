import numpy as np
from PIL import Image
import os, sys
import torch
import torchvision
import torchvision.transforms as transform
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import csv


#Transform the data
Norm = transform.Compose([transform.Resize((334,334)),transform.ToTensor(), transform.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
Gray = transform.Grayscale(num_output_channels=3)
To_tensor = transform.ToTensor()
#Get the dict
Landmarks = (pd.read_csv("../train.csv")).values
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
#open the submission
Submission = (pd.read_csv("../sample_submission.csv")).values
#Prepare the model
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(12800,5005)
resnet18 = resnet18.eval()
resnet18.load_state_dict(torch.load('../Saved_models/No_aug/resnet18.pkl'))
#Saving
Pic = []
Prob = []
path = "../Test"
dirs = os.listdir( path )
for j, i in enumerate(dirs):

    #name = Submission[j,0]
    #predict = Submission[j,1]
    Submission[j,0] = str(i)
    #Open the image
    img = Image.open("../Test/" + str(i))
    #Adding layer to one layer image
    x = To_tensor(img)
    if x.size()[0] == 1:
        img = Gray(img)
    #Resize the image
    img = Norm(img)
    img = img.view(1,3,334,334)
    output = resnet18(img)
    index = torch.topk(output,5)
    index = index[1]
    index = index.numpy()
    index = index[0]
    new_prob = []
    for s in range(len(index)):
        for key, value in dict.items():
            if value == index[s]:
                new_prob.append(key)
    Submission[j,1] = np.asarray(new_prob)
    if j % 100 == 0:
        print(j)

foo = pd.DataFrame(Submission)
foo.to_csv("../Sub_No_aug.csv")
