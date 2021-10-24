# Import needed packages
import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
from torchvision import models
import numpy as np


def load_model(model_path):
  model= models.resnet50(pretrained=True)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 2)
 
  model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
  model.eval()
  return model
# Load your model to this variable
model = load_model('D:/Dev Projects/AI_Projects/Blood Smear Code/notebooks/smear_analyser.pt')


def predict_image(img_path):
  print('Image prediction in progress')
  image=Image.open(img_path)

  transformation=transforms.Compose(
      [
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
      ]
  )
  img_tensor=transformation(image).float()
  img_tensor=img_tensor.unsqueeze_(0)

  if torch.cuda.is_available():
    img_tensor.cuda()
  
  from torch.autograd import variable
  input=variable(img_tensor)
  output=model(input)

  idx=output.data.numpy().argmax()
  return idx


if __name__ == "__main__":

    imagefile = "D:/Dev Projects/AI_Projects/Blood Smear Code/cell_images/Test/Uninfected/C1_thinF_IMG_20150604_104722_cell_164.png"
    imagepath = os.path.join(os.getcwd(), imagefile)
   
    # run prediction function annd obtain prediccted class index
    index = predict_image(imagepath)
    if(index==0):
        print("Postive malaria")
    elif(index==1):
        print("Negative")

   
#D:/Dev Projects/AI_Projects/Blood Smear Code/cell_images/Test/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_181.png
#D:/Dev Projects/AI_Projects/Blood Smear Code/cell_images/Test/Uninfected/C1_thinF_IMG_20150604_104722_cell_164.png