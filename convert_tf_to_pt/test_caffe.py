
import sys
import torch
import caffe
from PIL import Image
from torchvision import transforms

if not len(sys.argv) == 2:
  sys.exit("too less argc : model name is needed! -> efficientnet-b0/efficientnet-b1/efficientnet-b2/efficientnet-b3")

model_name = sys.argv[1]
deploy = "caffemodel/" + model_name + ".prototxt"
caffemodel = "caffemodel/" + model_name + ".caffemodel"

tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('test.jpg')).unsqueeze(0).numpy()

model = caffe.Net(deploy,caffemodel,caffe.TEST)
model.blobs['data'].data[...] = img
out = model.forward()
prob= model.blobs["_fc"].data
print prob.reshape(-1)
print img.shape
