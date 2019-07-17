from __future__ import print_function
import json
import torch
import numpy as np
from PIL import Image
from model import *
from torchvision import transforms

model = get_from_pretrained('efficientnet-b0')

tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('test.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])
print(img.dtype)

model.eval()
with torch.no_grad():
  outputs = model(img)

print(outputs.reshape(-1).astype(np.float32))
print(outputs.shape)
#print(type(outputs.numpy()))

#dict = torch.load("../pretrained_pytorch/efficientnet-b0.pth")
#print(dict["_conv_stem.weight"])