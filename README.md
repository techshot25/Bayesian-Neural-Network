

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:42:44 2019

Adopted from: https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd

@author: Ali Shannon
"""
#%% import useful modules



from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro
import pyro.distributions as dist

import torch.nn.functional as F
import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

pyro.set_rng_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% Make a classifier

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc = nn.Linear(28*28, 256)
        self.out = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.out(x)
    
net = Net().to(device)


#%% Lift into a pyro module

def model(x, y):
    priors = {}
    for name, param in net.named_parameters():
        priors[name] = dist.Normal(loc=torch.zeros_like(param), scale=torch.ones_like(param))
    
    lifted_module = pyro.random_module('module', net, priors)
    lifted_clf_module = lifted_module()
    
    lhat = F.log_softmax(lifted_clf_module(x), -1)
    
    pyro.sample('obs', dist.Categorical(logits=lhat), obs=y)
    

#%%
    
def guide(x, y):
    priors = {}
    for name, param in net.named_parameters():
        mu = torch.randn_like(param)
        sigma = torch.randn_like(param)
        mu_param = pyro.param(name+'_mu', mu)
        sigma_param = F.softplus(pyro.param(name+'_sigma', sigma))
        prior = dist.Normal(loc=mu_param, scale=sigma_param)
        priors[name] = prior
        
    lifted_module = pyro.random_module('module', net, priors)
    return lifted_module()

#%% Define error and optimizer
    
opt = Adam({'lr': 0.01})
svi = SVI(model, guide, opt, loss=Trace_ELBO())

#%% Get data

train_loader = DataLoader(datasets.MNIST('./mnist', train=True, 
                                         transform=transforms.ToTensor(),
                                         download=True),
                        batch_size=100, num_workers=4)

test_loader = DataLoader(datasets.MNIST('./mnist', train=False, 
                                         transform=transforms.ToTensor()),
                        batch_size=100, num_workers=4)
    
#%% Train the network

def train():
    pyro.clear_param_store()
    epochs = 10
    loss = 0
    for epoch in range(1, epochs + 1):
        loss = 0
        for x, y in train_loader:
            loss += svi.step(x.flatten(1).to(device), y.to(device))
        total_loss = loss / len(train_loader.dataset)

        print(f'epoch: {epoch}\tLoss: {total_loss:.4g}')
        
train()
```

    epoch: 1	Loss: 536.7
    epoch: 2	Loss: 67.28
    epoch: 3	Loss: 38.71
    epoch: 4	Loss: 34.65
    epoch: 5	Loss: 34.06
    epoch: 6	Loss: 34.13
    epoch: 7	Loss: 34.21
    epoch: 8	Loss: 34.16
    epoch: 9	Loss: 34.23
    epoch: 10	Loss: 34.39



```python
#%% Evaluate the network

num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x.to(device)).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.cpu().numpy(), axis=1)

print('Prediction when network is forced to predict')
correct = 0
total = 0
for x, y in test_loader:
    predicted = predict(x.flatten(1))
    total += y.size(0)
    correct += (predicted == y.numpy()).sum().item()
print(f'Accuracy: {correct/total:.2%}')
```

    Prediction when network is forced to predict
    Accuracy: 85.38%

