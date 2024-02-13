import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net, args_input, args_task,loader):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task
        self.loader=loader
    def query(self, n):
        pass
    
    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)
    
    def get_model(self):
        return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        #print(self.dataset.labeled_idxs)
        self.dataset.labeled_idxs[pos_idxs] = True
        #print(self.dataset.labeled_idxs)
        #print(pos_idxs)
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, data = None, model_name = None):
        #print(model_name,data)
        if model_name == None:
            if data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
                #print("seee",len(labeled_data),len(labeled_idxs),len(unlabeled_idxs),len(unlabeled_data))
                print(labeled_data.X.shape[1:],labeled_data.X.shape)
                self.net.train(labeled_data)
            else:
                self.net.train(data)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def train_1(self,x,y):
        self.net.train_1(x,y)
        #return output
    
    def train_2(self,dataset,net,args_input,args_task,NUM_QUERY,unlabeled_idxs,loader):
        self.net.train_2(dataset,net,args_input,args_task,NUM_QUERY,unlabeled_idxs,loader)
        #return idxs
    def predict1(self, dataloader):
        preds=self.net.predict1(dataloader)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob1(self, data):
        probs = self.net.predict_prob1(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split1(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split1(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

    def get_embeddings1(self, data):
        embeddings = self.net.get_embeddings1(data)
        return embeddings
    
    def get_grad_embeddings(self, data):
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings

    def get_grad_embeddings1(self, data):
        embeddings = self.net.get_grad_embeddings1(data)
        return embeddings

