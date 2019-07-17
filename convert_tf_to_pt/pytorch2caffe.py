
import torch
import torchvision

import os
import sys
from model_caffe import *

if not len(sys.argv) == 2:
  sys.exit("too less argc : model name is needed! -> efficientnet-b0/efficientnet-b1/efficientnet-b2/efficientnet-b3")

model_name = sys.argv[1]
model = get_from_name(model_name) #from_pretrained("efficientnet-b0")#torch.load("efficientnet-b0")
model.cpu()
model.eval()
print(model)
#print(model.state_dict())
#print(m)

#input_var = Variable(torch.rand(1, 3, 224, 224))
#output_var = model(input_var)
#print(output_var)
#output_dir = './'
# plot graph to png
#plot_graph(output_var, os.path.join(output_dir, 'inception_v3.dot'))
#pytorch2caffe(input_var, output_var, 'efficientnet-b0.prototxt','efficientnet-b0.caffemodel')
