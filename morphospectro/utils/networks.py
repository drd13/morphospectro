"""boillerplate pytorch nn that I borrowed from some other project"""
import torch.nn as nn
import torch
import torch.nn.functional as F




class Feedforward(nn.Module):
    def __init__(self,structure,activation=nn.LeakyReLU(),with_dropout=False,final_activation=False):
        super(Feedforward, self).__init__()        
        self.layers = []
        for i in range(len(structure)-2):
            if i==1 and with_dropout==True:
              print("using dropout")
              self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.Linear(structure[i],structure[i+1]))
            self.layers.append(activation)
        
        self.layers.append(nn.Linear(structure[-2],structure[-1]))
        if final_activation is True:
            self.layers.append(activation)
            
        self.fc = nn.Sequential(*self.layers)


    def forward(self, x,optional=False,optional2=False):
        #optional was added so that feedforward took as many inputs as embedding_decoder
        #for layer in layers:
        #    x = layer(x)
        x = self.fc(x)
        return x
    
    
class FullNet(nn.Module):
    def __init__(self,spec_network,im_network,merged_network):
        super(FullNet, self).__init__()      
        self.im_network = im_network
        self.spec_network = spec_network
        self.merged_network = merged_network


    def forward(self, spec, im):
        latent_im  = self.im_network(im)
        latent_spec  = self.spec_network(spec)
        #merged_latent = latent_im-latent_spec
        #pred_match = self.merged_network(merged_latent)
        #print(latent_im.shape)
        merged_latent = torch.cat((latent_im,latent_spec),dim=1)
        #print(merged_latent.shape)
        pred_match = self.merged_network(merged_latent)
        return pred_match
    
    
class ConvNetBasic(nn.Module):

    def __init__(self,n_channels = 3):
        super(ConvNetBasic, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(n_channels, 12, 3)
        self.conv2 = nn.Conv2d(12, 24, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(864, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 50)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
    

    
    
class ConvNet(nn.Module):
    def __init__(self,n_channels=3):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(n_channels, 8,4,stride=2)
        self.conv2 = nn.Conv2d(8, 16, 4,stride=1)
        self.conv3 = nn.Conv2d(16, 32, 4,stride=1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(5408, 512)  # 6*6 from image dimension
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 50)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    

    
    
    

class SpectraEmbedding(nn.Module):
    def __init__(self,ndf):
        super(SpectraEmbedding,self).__init__()
        nc=1
        # 4096
        self.layer1 = nn.Sequential(nn.Conv1d(nc,ndf,kernel_size=10,stride=4,padding=3),
                                 nn.BatchNorm1d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 1024
        self.layer2 = nn.Sequential(nn.Conv1d(ndf,ndf*2,kernel_size=10,stride=4,padding=3),
                                 nn.BatchNorm1d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 256
        self.layer3 = nn.Sequential(nn.Conv1d(ndf*2,ndf*4,kernel_size=10,stride=4,padding=3),
                                 nn.BatchNorm1d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 64
        self.layer4 = nn.Sequential(nn.Conv1d(ndf*4,ndf*4,kernel_size=10,stride=4,padding=3),
                                 nn.BatchNorm1d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        #16
        self.layer5 = nn.Sequential(nn.Conv1d(ndf*4,16,kernel_size=16,stride=1,padding=0),
                                 nn.LeakyReLU(0.2,inplace=True))


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out



class ImageEmbedding(nn.Module):
    def __init__(self,ndf):
        super(ImageEmbedding,self).__init__()
        nc=3
        # 32 x 32
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 8 x 8
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 4 x 4
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,16,kernel_size=4,stride=1,padding=0),
                                 nn.LeakyReLU(0.2,inplace=True))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    

    
