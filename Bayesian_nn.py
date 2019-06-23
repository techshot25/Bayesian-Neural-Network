#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:42:44 2019

# https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd

@author: ali
"""
#%% import useful modules


# pip install pyro-ppl
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro
import pyro.distributions as dist


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import torch.nn.functional as F
import torch
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
    epochs = 7
    loss = 0
    for epoch in range(1, epochs + 1):
        loss = 0
        for x, y in train_loader:
            loss += svi.step(x.flatten(1).to(device), y.to(device))
        total_loss = loss / len(train_loader.dataset)

        print(f'epoch: {epoch}\tLoss: {total_loss:.4g}')
        
train()

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



#%%

classes = [str(i) for i in range(10)]

num_samples = 100
def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.flatten(1).to(device)).data, 1).cpu().detach().numpy() for model in sampled_models]
    return np.asarray(yhats)
    #mean = torch.mean(torch.stack(yhats), 0)
    #return np.argmax(mean, axis=1)


def test_batch(images, labels, plot=True):
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions=0

    for i in range(len(labels)):
    
        if(plot):
            print("Real: ",labels[i].item())
            fig, axs = plt.subplots(1, 10, sharey=True,figsize=(20,2))
    
        all_digits_prob = []
    
        highted_something = False
    
        for j in range(len(classes)):
        
            highlight=False
        
            histo = []
            histo_exp = []
        
            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))
            
            prob = np.percentile(histo_exp, 50) #sampling median probability
        
            if(prob>0.2): #select if network thinks this sample is 20% chance of this being a label
                highlight = True #possibly an answer
        
            all_digits_prob.append(prob)
            
            if(plot):
            
                N, bins, patches = axs[j].hist(histo, bins=8, color = "lightgray", lw=0,  weights=np.ones(len(histo)) / len(histo), density=False)
                axs[j].set_title(str(j)+" ("+str(round(prob,2))+")") 
        
            if(highlight):
            
                highted_something = True
                
                if(plot):

                    # We'll color code by height, but you could use any scalar
                    fracs = N / N.max()

                    # we need to normalize the data to 0..1 for the full range of the colormap
                    norm = colors.Normalize(fracs.min(), fracs.max())

                    # Now, we'll loop through our objects and set the color of each accordingly
                    for thisfrac, thispatch in zip(fracs, patches):
                        color = plt.cm.viridis(norm(thisfrac))
                        thispatch.set_facecolor(color)

    
        if(plot):
            plt.show()
    
        predicted = np.argmax(all_digits_prob)
    
        if(highted_something):
            predicted_for_images+=1
            if(labels[i].item()==predicted):
                if(plot):
                    print("Correct")
                correct_predictions +=1.0
            else:
                if(plot):
                    print("Incorrect :()")
        else:
            if(plot):
                print("Undecided.")
        
        if(plot):
            plt.imshow(images[i].squeeze())
        
    
    if(plot):
        print("Summary")
        print("Total images: ",len(labels))
        print("Predicted for: ",predicted_for_images)
        print("Accuracy when predicted: ",correct_predictions/predicted_for_images)
        
    return len(labels), correct_predictions, predicted_for_images



#%% Prediction when network can decide not to predict

print('Prediction when network can refuse')
correct = 0
total = 0
total_predicted_for = 0
for j, data in enumerate(test_loader):
    images, labels = data
    
    total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels, plot=False)
    total += total_minibatch
    correct += correct_minibatch
    total_predicted_for += predictions_minibatch

print("Total images: ", total)
print("Skipped: ", total-total_predicted_for)
print("Accuracy when made predictions: %d %%" % (100 * correct / total_predicted_for))

#%% preview

dataiter = iter(test_loader)
images, labels = dataiter.next()

test_batch(images[:100], labels[:100])
