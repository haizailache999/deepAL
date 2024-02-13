import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init

class AL_Net(nn.Module):
	def __init__(self, dim = 32 * 32*3, pretrained=False, num_classes = 2):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.softmax=torch.nn.Softmax(dim=1)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))
		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.softmax(self.classifier(x))	
		return output
	
	def get_embedding_dim(self):
		return self.dim
	
class Architect(object):
    def __init__(self, model):
        self.model = model
        self.model.train()
        self.optimizer = torch.optim.Adam([{'params':[ param for name, param in model.named_parameters() if '_1' in name]}],
            lr=0.0005, betas=(0.5, 0.999), weight_decay=0.1)
        
    def step(self, input_valid, target_valid,device,score):
        self.optimizer.zero_grad()
        #for name, parms in self.model.named_parameters():	
            #print(name)
        self._backward_step(input_valid.to(device), target_valid.to(device),score,device)
        self.optimizer.step()
        #for name, parms in self.model.named_parameters():	
            #print('after-->name:', name)
            #print('after-->para:', parms)
            #print('after-->grad_requirs:',parms.requires_grad)
            #if parms.grad is not None:
                #print('after-->grad_value:',name)
            #print("after===")
            #break

    def _backward_step(self, input_valid, target_valid,score,device):
        loss = F.cross_entropy(input_valid, target_valid,reduction='none')
        n=np.sum(score == 1)
        if n>128*0.05:
            add_loss=n-128*0.05
        else:
            add_loss=0
        score=torch.from_numpy(score).requires_grad_(loss.requires_grad).to(device)
        print(add_loss)
        loss=(loss*score).mean()+add_loss
        print(loss)
        #loss=Variable(loss, requires_grad=True)
        loss.backward()