import torch
import torch.nn as nn
import torchvision.models as models

resnet101 = models.resnet101(pretrained=True)
resnet101.fc = nn.Linear(2048,5004)
resnet101.load_state_dict(torch.load('./Saved_models/resnet101.pkl'))
