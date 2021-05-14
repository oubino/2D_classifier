# model

import torch
import torch.nn as nn

s = 16


class classifier(nn.Module):
    
  def __init__(self):
    super(classifier,self).__init__()
    
    self.pool = nn.MaxPool2d(3,3) # define pool as max 2x2
    
    self.dropout1 = nn.Dropout2d(p=0.2) # spatial dropout
    self.dropout2 = nn.Dropout2d(p=0.5)

    self.relu = nn.LeakyReLU()
    self.softmax = nn.Softmax()
    self.swish = nn.Hardswish()
    #self.adpool = nn.AdaptiveAvgPool2d(((1,1)))

    self.conv1 = nn.Conv2d(2,s,3,padding =1) # i.e. input channel, output channels, Kernel size
    self.batch1 = nn.BatchNorm2d(s) # batch normalisation

    self.conv2 = nn.Conv2d(s,2*s,3, padding = 1)
    self.batch2 = nn.BatchNorm2d(2*s)

    self.conv3 = nn.Conv2d(2*s,4*s,3, padding = 1)
    self.batch3 = nn.BatchNorm2d(4*s)

    self.conv4 = nn.Conv2d(4*s,8*s,3, padding = 1)
    self.batch4 = nn.BatchNorm2d(8*s)

    self.adpool = nn.AdaptiveAvgPool2d((1,1))
    self.conv5 == nn.Conv2d(8*s,2,3,padding = 1)


    self.fc1 = nn.Linear(in_features = 8*s*1*1, out_features = 64) # pool
    self.fc2 = nn.Linear(in_features = 64, out_features = 16)
    self.out = nn.Linear(in_features = 16, out_features = 2)
  
  def forward(self,x):
    x = self.relu(self.pool(self.batch1(self.conv1(x))))
    x = self.relu(self.pool(self.batch2(self.conv2(x))))
    x = self.relu(self.pool(self.batch3(self.conv3(x))))
    x = self.relu(self.pool(self.batch4(self.conv4(x))))
    x = self.dropout1(x)
    print(x.shape)
    x = self.relu(self.pool(self.batch4(self.conv5(x))))
    print(x.shape)
    x = self.adpool(x)
    print(x.shape)
    
    
    torch.flatten(x, 1)
    x = x.view(-1,8*s*1*1)
    #x = x.view(settings.batch_size,128)  
    
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.out(x)
    
    return x
