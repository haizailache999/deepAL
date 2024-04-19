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
import math
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
        #if '_1' in name
        self.optimizer = torch.optim.Adam([{'params':[ param for name, param in model.named_parameters()]}],
            lr=0.0005, betas=(0.5, 0.999), weight_decay=0.1)
        
    def step(self, input_valid, target_valid,device,score,length):
        self.optimizer.zero_grad()
        #for name, parms in self.model.named_parameters():	
            #print(name)
        loss,loss1,loss2=self._backward_step(input_valid.to(device), target_valid.to(device),score,device,length)
        self.optimizer.step()
        '''for name, parms in self.model.named_parameters():	
            #print('after-->name:', name)
            #print('after-->para:', parms)
            #print('after-->grad_requirs:',parms.requires_grad)
            if parms.grad is not None:
                if "_1" in name:
                    print('after-->grad_value:',name)
                    print(parms.grad)
            #print("after===")
            #break'''
        return loss,loss1,loss2

    def _backward_step(self, input_valid, target_valid,score,device,length):
        loss = F.cross_entropy(input_valid, target_valid,reduction='none')
        loss_test=F.cross_entropy(input_valid, target_valid,reduction='mean')
        file = open("./loss_check9.txt", 'a')
        #file.write("loss:")
        file.write(str(loss_test.item()))
        file.write("  ")
        n=torch.sum(score < 0.5).item()
        #print("this is n",n,length*0.02)
        if n>length*0.02:
            add_loss=(1/(1+math.exp(0.5*(length*0.02-n)))-0.5)*2
        else:
            add_loss=(1/(1+math.exp(0.5*(n-length*0.02)))-0.5)*2
        #score=torch.from_numpy(score).requires_grad_(True).to(device)
        #print("add",add_loss)
        #print(loss.device,score.device)
        #print(score.requires_grad)
        #print(score)
        #print(loss*score)
        loss1=(loss*score).mean()
        loss2=add_loss
        #loss2=0
        loss=loss1
        #loss=score.mean()
        #print("loss",loss)
        #file = open("./check_loss.txt", 'a')
        #file.write("loss:")
        #loss=Variable(loss, requires_grad=True)
        #print(loss.requires_grad)
        #print(loss)
        #with torch.autograd.detect_anomaly():
        loss.backward()
        return loss,loss1,loss2
