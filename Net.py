# import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_pooling(pool):
    if pool == 1:
        pooling=nn.MaxPool2d

    else:
        pooling=nn.AvgPool2d
    return pooling

class Net(nn.Module):
  def __init__(self, num_layer, kernel_sizes, num_filters, pooling, pooling_size, mid_units):
    super(Net, self).__init__()
    self.activation = F.relu
    
    #first layer
    self.pads = nn.ModuleList([nn.ZeroPad2d((int(kernel_sizes[0]/2),(int(kernel_sizes[0]/2) if kernel_sizes[0] %2 ==1 else int(kernel_sizes[0]/2)-1),0,0))])
    self.convs = nn.ModuleList([nn.Conv2d(in_channels=155, 
                                          out_channels=num_filters[0],
                                          kernel_size=(1,kernel_sizes[0]),
#                                           padding=(0,0,0,int(kernel_sizes[0]/2)),
                                          stride=(1,1))])
#     pooling_size = [1 for i in range(num_layer)]
#     pooling_size[0] = 4
#     pooling_size[2] = 3
    self.poolings = nn.ModuleList([])
    
    
    self.out_height = 1
    self.out_width = 14
    self.bnorms = nn.ModuleList([nn.BatchNorm2d(num_filters[0])])
    
  #mid unit
    for i in range(1, num_layer):
      self.pads.append(nn.ZeroPad2d((int(kernel_sizes[i]/2),(int(kernel_sizes[i]/2) if kernel_sizes[i] %2 ==1 else int(kernel_sizes[i]/2)-1),0,0)))
      self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=(1,kernel_sizes[i]),
#                                   padding=(0,int(kernel_sizes[i]/2)),
                                  stride=(1,1)))
      self.out_width = int(self.out_width /pooling_size[i-1])
      self.poolings.append(get_pooling(pooling[i-1])(kernel_size=(1,pooling_size[i-1]), stride=(1,pooling_size[i-1]), padding=(0,0)))
      self.bnorms.append(nn.BatchNorm2d(num_filters[i]))

    #pooling layer
    
    # pooling_sizes.append(get_pool_size(self.out_width, i))
    self.poolings.append(get_pooling(pooling[-1])(kernel_size=(1,self.out_width), stride=(1,2), padding=(0,0)))

#     self.pool = nn.BatchNorm2d(num_filters[-1])
#     self.out_height = 1
#     self.out_width = self.out_width
    #linear layer
    self.out_feature = num_filters[num_layer - 1]
    self.fc1 = nn.Linear(in_features=self.out_feature, out_features=mid_units) 
    # self.fc2 = nn.Linear(in_features=mid_units, out_features=100) 
    self.fc3 = nn.Linear(in_features=mid_units, out_features=1)

  def forward(self, x):
#     print(x.shape,self.out_width1,self.out_width2)
    
    for i, l in enumerate(self.convs):

      x = self.pads[i](x)
      x = l(x)
      x = self.poolings[i](x)
      x = self.bnorms[i](x)
#       print(x.shape)
      
      x = self.activation(x)
#       print(x.shape,self.out_width,self.out_width2,self.out_width3,self.a)
#     x = self.pools[0](x)
    
#     x = x.view(x.size(0), self.out_feature)
#     print(x.shape)
#     print(x.shape,self.out_width)
    
    
    x = x.view(x.size(0), x.size(1) * x.size(2)*x.size(3)) 
#     print(x.shape)
#     x = x.unsqueeze_(12288 , 14)
    x = self.fc1(x)
    # x = self.fc2(x)
    x = self.fc3(x)
#     print(x)
    return x
  def num_flat_features(self,x):

    size=x.size()[1:]
    num_features=1
    for s in size:
        num_features*=s
    return num_features