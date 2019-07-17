
import sys
import torch
import caffe
import numpy as np
from torch.utils import model_zoo

if not len(sys.argv) == 2:
  sys.exit("too less argc : model name is needed! -> efficientnet-b0/efficientnet-b1/efficientnet-b2/efficientnet-b3")

model_name = sys.argv[1]

dict = torch.load("../pretrained_pytorch/" + model_name + ".pth")
model = caffe.Net("caffemodel/" + model_name + ".prototxt",caffe.TEST)

for kv in model.params:
  print kv
for k,v in dict.items():
  print k,v.numpy().shape

for k,v in dict.items():
  #batch normalization
  #print("PARAMETERS -> ",k)
  if "bn" in k:
    #print("Batch_Normalization -->> ",k)
    if ".weight" in k:
      caffe_key = k.replace(".weight","_scale")
      print (k," --> ",caffe_key,model.params[caffe_key][0].data.shape," --> ",v.numpy().shape)
      if model.params[caffe_key][0].data.shape != v.numpy().shape:
        sys.exit("data shape is not equal!")
      model.params[caffe_key][0].data.flat = v.numpy().flat
    elif ".bias" in k:
      caffe_key = k.replace(".bias","_scale")
      print (k," --> ",caffe_key,model.params[caffe_key][1].data.shape," --> ",v.numpy().shape)
      if model.params[caffe_key][1].data.shape != v.numpy().shape:
        sys.exit("data shape is not equal!")
      model.params[caffe_key][1].data.flat = v.numpy().flat
    elif ".running_mean" in k:
      caffe_key = k.replace(".running_mean","")
      print (k," --> ",caffe_key,model.params[caffe_key][0].data.shape," --> ",v.numpy().shape)
      if model.params[caffe_key][0].data.shape != v.numpy().shape:
        sys.exit("data shape is not equal!")
      model.params[caffe_key][0].data.flat = v.numpy().flat
      model.params[caffe_key][2].data[...] = 1
    elif ".running_var" in k:
      caffe_key = k.replace(".running_var","")
      print (k," --> ",caffe_key,model.params[caffe_key][1].data.shape," --> ",v.numpy().shape)
      if model.params[caffe_key][1].data.shape != v.numpy().shape:
        sys.exit("data shape is not equal!")
      model.params[caffe_key][1].data.flat = v.numpy().flat
      model.params[caffe_key][2].data[...] = 1
    else:
      print(k," parammeters parser error!")
  else:
    #print("ConV/FC -->> ",k)
    if ".weight" in k:
      caffe_key = k.replace('.weight','')
      print (k," --> ",caffe_key,model.params[caffe_key][0].data.shape," --> ",v.numpy().shape)
      if model.params[caffe_key][0].data.shape != v.numpy().shape:
        sys.exit("data shape is not equal!")
      model.params[caffe_key][0].data.flat = v.numpy().flat
    elif ".bias" in k:
      caffe_key = k.replace('.bias','')
      print (k," --> ",caffe_key,model.params[caffe_key][1].data.shape," --> ",v.numpy().shape)
      if model.params[caffe_key][1].data.shape != v.numpy().shape:
        sys.exit("data shape is not equal!")
      model.params[caffe_key][1].data.flat = v.numpy().flat
    else:
      print(k," parammeters parser error!")

model.save("caffemodel/" + model_name + ".caffemodel")
#for kv in model.params:
#  print kv

#print(type(dict))
#for k,v in dict.items():
#  print k
#print(dict.items())
#print dict


