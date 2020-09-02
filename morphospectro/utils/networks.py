"""boillerplate pytorch nn that I borrowed from some other project"""
import torch.nn as nn
import torch
import torch.nn.functional as F




class Feedforward(nn.Module):
    def __init__(self,structure,activation=nn.LeakyReLU(),with_dropout=False):
        super(Feedforward, self).__init__()        
        self.layers = []
        for i in range(len(structure)-2):
            if i==1 and with_dropout==True:
              print("using dropout")
              self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.Linear(structure[i],structure[i+1]))
            self.layers.append(activation)
        
        self.layers.append(nn.Linear(structure[-2],structure[-1]))
        self.fc = nn.Sequential(*self.layers)


    def forward(self, x,optional=False,optional2=False):
        #optional was added so that feedforward took as many inputs as embedding_decoder
        #for layer in layers:
        #    x = layer(x)
        x = self.fc(x)
        return x