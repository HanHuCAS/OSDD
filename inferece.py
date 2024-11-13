from models.convnext import convnext_base,convnext_large
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
import numpy as np 

model=convnext_base(num_classes=9).cuda()
ckp=torch.load('/home/xyzhou/code/aisc/traincode/convnext/result/result_base_n9/checkpoint-49.pth')
model.load_state_dict(ckp['model'])
model.eval()

transform=Compose([Resize((224,224)),ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset=ImageFolder('/old_home/xyzhou/aisc/images/train/u2l',transform=transform)
data_loader = DataLoader(dataset=dataset,batch_size=64,num_workers=4,shuffle=False,drop_last=False)

with torch.no_grad():
    for i,(img,t) in enumerate(data_loader):
        img=img.cuda()
        out=model(img)
        ot=np.array(torch.max(out,dim=1).indices.detach().cpu())
        print(ot)
