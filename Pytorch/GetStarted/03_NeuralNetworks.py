#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print('forward_maxpool2d_size: {}'.format(x.size()))
        x = x.view(-1, self.num_flat_featrues(x))
        print('flat: {}'.format(x.size()))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    def num_flat_featrues(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)

## The learnable parameters 
print('=====Parameters=====')
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's weight
print(params[1].size()) # conv1's bias
print(params[2].size())
print(params[3].size())
print(params[4].size())
print(params[5].size())


## Random try!
print('=====Random try!=====')
input = torch.rand(1, 1, 32, 32)
out = net(input)
print(out)
        

## Zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad() # 안해주면 메모리 터짐
out.backward(torch.rand(1, 10))
        
        
## Loss function
print('=====Loss function=====')
output = net(input)
target = torch.rand(10)
target = target.view(1, -1) # [10] --> [1,10]
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)


## Backprop
print('=====Backprop=====')
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bais.grad after backward')
print(net.conv1.bias.grad)


## Updated the weights, weight = weight - lr * gradient
print('=====Update the weigths=====')
# no1. python code
#lr = 0.01
#for f in net.parameters():
#    f.data.sub_(f.grad.data * lr)
    
# no2. optimizer 
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

# training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the update
        
        
        
    
    
    