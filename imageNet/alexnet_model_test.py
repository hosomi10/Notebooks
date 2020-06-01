from __future__ import print_function
#import matplotlib.pyplot as plt
#import torch.optim as optim
#from torchvision import datasets, transforms
import torch
from PIL import Image
from torchvision import transforms
from alexnet import alexnet

filename= 'data/images/polar_bear.jpg'

model = alexnet(pretrained=True)
#device = torch.device('cpu')
#model = model.to(device)

model.eval()

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output.shape)
#print(torch.nn.functional.softmax(output[0], dim=0))
with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100

_, indices = torch.sort(output, descending=True)
[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]