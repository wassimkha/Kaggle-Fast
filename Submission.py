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
Norm = transform.Compose([transform.Resize((224,224)),transform.ToTensor(), transform.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
Gray = transform.Grayscale(num_output_channels=3)
To_tensor = transform.ToTensor()
#Get the dict
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
#open the submission
Submission = (pd.read_csv("./Data/sample_submission.csv")).values
#Prepare the model
resnet18 = models.resnet101(pretrained=False)
resnet18.fc = nn.Linear(2048,5004)
resnet18.load_state_dict(torch.load('resnet101.pkl'))
resnet18 = resnet18.eval()

#Saving
Pic = []
Prob = []
path = "./Data/Images/Test"
dirs = os.listdir( path )
for j, i in enumerate(dirs):

    #name = Submission[j,0]
    #predict = Submission[j,1]
    Submission[j,0] = str(i)
    #Open the image
    img = Image.open("./Data/Images/Test/" + str(i))
    #Adding layer to one layer image
    img = Gray(img)
    #Resize the image
    img = Norm(img)
    img = img.view(1,3,224,224)
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
foo.to_csv("./101Test.csv")
