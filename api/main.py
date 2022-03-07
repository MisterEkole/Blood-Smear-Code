import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
from torchvision import models
import numpy as np

from fastapi import FastAPI, UploadFile, File
import uvicorn

app=FastAPI()



def load_model(model_path):
    model=models.resnet50(pretrained=True)
    num_fts=model.fc.in_features
    model.fc=nn.Linear(num_fts,2)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval
    
    return model

model=load_model('D:/Dev Projects/AI_Projects/Blood Smear Code/notebooks/smear_analyser.pt')

@app.get('/prediction')

async def predict_image(img_path):
    image=Image.open(img_path)
    
    transformation= transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            tranforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            
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
    



    
