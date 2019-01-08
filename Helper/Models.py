import torch
import torch.nn as nn
import torchvision.models as models

resnet152 = models.resnet152(pretrained=True)
 #To change
# resnet152.load_state_dict(torch.load('./Saved_models/resnet152.pkl'))
