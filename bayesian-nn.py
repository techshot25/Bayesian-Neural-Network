#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:47:23 2019

Credit:
    https://github.com/paraschopra/bayesian-neural-network-mnist

@author: ashannon
"""

#%% Libraries to use

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from matplotlib import colors

import pyro # pip install pyro-ppl
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% Define the PyTorch module

class NN(nn.Module):
    def __init__(self, in_shape, hidden_shape, out_shape):
        super(NN, self).__init__()
        
        self.fc1 = nn.Linear(in_shape, hidden_shape)
        self.out = nn.Linear(hidden_shape, out_shape)
        
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.out(x)
        
    
train_loader = DataLoader(
        datasets.MNIST('emnist', train=True, download=True,
                       transform=transforms.ToTensor()),
                batch_size=128, shuffle=True, num_workers=4)
        
test_loader = DataLoader( 
        datasets.MNIST('emnist', train=False,
                       transform=transforms.ToTensor()),
                batch_size=128, shuffle=True, num_workers=4)
        
# this will be our neural network
net = NN(28*28, 1024, 10).to(device)

#%% Convert to Pyro model

def model(x, y):
    priors = {}
    for name, param in net.named_parameters():
        priors[name] = dist.Normal(loc=torch.zeros_like(param.data), 
                                        scale=torch.ones_like(param.data))
        
    lifted_module = pyro.random_module("module", net, priors)
    lifted_clf_model = lifted_module()
    lhat = F.log_softmax(lifted_clf_model(x), 1)
    pyro.sample("obs", dist.Categorical(logits=lhat), obs=y)
    
#%% Define the guide for SVI
    
def guide(x, y):
    priors = {}
    for name, param in net.named_parameters():
        mu = torch.randn_like(param)
        sigma = torch.randn_like(param)
        mu_param = pyro.param(name+'.mu', mu)
        sigma_param = F.softplus(pyro.param(name+'.sigma', sigma))
        prior = dist.Normal(loc=mu_param, scale=sigma_param)
        priors[name] = prior
        
    lifted_module = pyro.random_module('module', net, priors)
    return lifted_module()

#%% Set up optimizer and loss function

opt = optim.Adam({'lr': 0.01})
svi = SVI(model, guide, opt, loss=Trace_ELBO())

#%% Train Model

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
    
#%% Evaluate model
"""There are two ways to evaluate this network.
The first is to force it to predict even if it's unsure.
The second is to make the network say 'I don't know.' Which is what we want
Let's try the first one:
"""
    
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

#%% Now we let the network decide whether or not to predict 

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
        print(f"Accuracy when predicted: {correct_predictions/predicted_for_images:.2%}")
        
    return len(labels), correct_predictions, predicted_for_images



#%% Prediction when network can decide not to predict

print('Prediction when network can refuse')
correct = 0
total = 0
total_predicted_for = 0
for x, y in test_loader:
    
    total_minibatch, correct_minibatch, predictions_minibatch = test_batch(
                                    x.to(device), y.to(device), plot=False)
    total += total_minibatch
    correct += correct_minibatch
    total_predicted_for += predictions_minibatch

print("Total images: ", total)
print("Skipped: ", total-total_predicted_for)
print(f"Accuracy when made predictions: {correct/total_predicted_for:.2%}")
