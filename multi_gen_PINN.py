#import tensorflow as tf

import numpy as np
import torch
import torch.autograd as autograd   # computation graph
from torch.autograd import Variable
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import datetime, os
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import time
#from pyDOE import lhs         #Latin Hypercube Sampling
import os
from scipy.stats import qmc
from torch.func import functional_call, vmap, vjp, jvp, jacrev


def create_directory(dir):
    # If folder doesn't exist, then create it.
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print('Folder ' + dir + ' does not exist, so I made it for you! :)')
    else:
        print("Folder : " + dir + ' - CHECK!' )


#######################
#### TRAINING DATA ####
#######################

def trainingdata(N_f, lb, ub, u_sol, dynamical_system = False):
    
    #n_points = 5
    #x = np.linspace(0.0,1.0,n_points)

    #x_new = x[x<=0.75*ub]


    #X_train_data = x_new.reshape(x_new.shape[0], 1)
    #u_train_data = u_sol(x_new).reshape(x_new.shape[0], 1)
    

    '''Boundary Conditions'''

    #Left egde (x = -π and u = 0)
    leftedge_x = lb
    leftedge_u = u_sol(lb)
    
    #Right egde (x = π and u = 0)
    rightedge_x = ub
    rightedge_u = u_sol(ub)
    
    X_u_train = np.vstack([leftedge_x, rightedge_x])  # X_u_train [2,1]
    u_train = np.vstack([leftedge_u, rightedge_u])  #corresponding u [2x1]

    if dynamical_system :
        X_u_train = X_u_train[0,:][:,None]
        #X_u_train = np.vstack([X_u_train, X_train_data])  # X_u_train [2,1]
        
        u_train = u_train[0,:][:,None]
        #u_train = np.vstack([u_train, u_train_data])

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    sampler = qmc.LatinHypercube(1)
    sample = sampler.random(N_f)
    X_f_train = scipy.stats.qmc.scale(sample, lb, ub)
    #X_f_train = lb + (ub-lb)*lhs(1,N_f)
    #X_f_train = np.linspace(lb,ub,n_points)
    X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points

    return X_f_train, X_u_train, u_train 


class NN_Poisson(nn.Module): 
    def __init__(self, layers, activation, learning_rate = 0.001):
        super().__init__()


        self.learning_rate = learning_rate
       
        self.layers = layers #Shape of the (hidden) layers      
        self.activation = activation #Activation function
        self.loss_function = nn.MSELoss(reduction ='mean') #Loss function for singular evaluation
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data,gain =1)
            nn.init.zeros_(self.linears[i].bias.data)    
            
        self.optimizer = torch.optim.LBFGS(self.parameters(), 0.1, 
                                      max_iter = 20, 
                                      max_eval = None, 
                                      tolerance_grad = 1e-11, 
                                      tolerance_change = 1e-11, 
                                      history_size = 100, 
                                      line_search_fn = 'strong_wolfe')
           
        #self.optimizer_Adam = torch.optim.Adam(self.parameters())
        self.optimizer_Adam = torch.optim.Adam(self.parameters(),  lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_SGD = torch.optim.SGD(self.parameters(), lr=0.01)
        
    def source(self, x):
        
        s = torch.zeros_like(x)
        for k in range(1,6):
            s += (2*k) * torch.sin(2*k*x) 
        return s
            
        
    def forward(self, x):  
        #x = (x+1)/2
        for i in range(len(self.layers)-2):
            x = self.activation(self.linears[i](x))
        out = self.linears[-1](x)
        return out
    
    def loss(self, xt_eq, xt_lr, xt_ic):         
        res1, res2 = self.residuals(xt_eq, xt_lr, xt_ic)
        return torch.sum(res1**2)+torch.sum(res2**2)
    
    def residuals(self, xt_eq, xt_lr, xt_ic):
        # Interior PDE
        xx = Variable(xt_eq, requires_grad=True)

        u = self(xx)
        
        f = self.source(xt_eq)

        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
         
        res1 = 1/np.sqrt(xt_eq.shape[0])*(u_xx + f)
       
        # LR and IC
        res2 = 1/np.sqrt(xt_lr.shape[0])*(self(xt_lr) - xt_ic)

        return res1, res2
    
    def res_forward(self, xt_eq, xt_lr, xt_ic):
        
        xx = Variable(xt_eq, requires_grad=True)
        
        #print(xt.shape)
        u = self(xx)

        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        
        res1 = 1/np.sqrt(xt_eq.shape[0])*u_xx
        res2 = 1/np.sqrt(xt_lr.shape[0])*self(xt_lr)

        return res1, res2
    
    def J_factory(self, xt_eq, xt_lr, xt_ic):
        # Detaching the parameters because we won't be calling Tensor.backward().
        params = {k: v.detach() for k, v in self.named_parameters()}
        def res_single(params, x):
            
            x=x.unsqueeze(0)
            f = self.source(x)
            xx = Variable(x, requires_grad=True)

            u = functional_call(self, params, xx) #SERVE LA FUCNTIONAL CALL!

            u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
            u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 

            res = u_xx + f
            return res

        def bc_lr_single(params, x, u):
            return functional_call(self, params, x) - u #SERVE LA FUCNTIONAL CALL!

        # Compute J(x1)
        jac1 = vmap(jacrev(res_single), (None, 0))(params, xt_eq)
        jac1 = 1/np.sqrt(xt_eq.shape[0])*torch.cat([jac1[j].flatten(2) for j in jac1],dim=2)

        jac2 = vmap(jacrev(bc_lr_single), (None, 0, 0))(params, xt_lr, xt_ic)
        jac2 =  1/np.sqrt(xt_lr.shape[0])*torch.cat([jac2[j].flatten(2) for j in jac2],dim=2)

        jac=torch.cat([jac1,jac2], dim=0)
        return jac.squeeze()




class NN_Oscillator(nn.Module): 
    def __init__(self, layers, activation, mu, k, learning_rate = 0.001):
        super().__init__()


        self.learning_rate = learning_rate
        self.mu, self.k = mu, k
       
        self.layers = layers #Shape of the (hidden) layers      
        self.activation = activation #Activation function
        self.loss_function = nn.MSELoss(reduction ='mean') #Loss function for singular evaluation
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data,gain =1)
            nn.init.zeros_(self.linears[i].bias.data)    
            
        self.optimizer = torch.optim.LBFGS(self.parameters(), 0.1, 
                                      max_iter = 20, 
                                      max_eval = None, 
                                      tolerance_grad = 1e-11, 
                                      tolerance_change = 1e-11, 
                                      history_size = 100, 
                                      line_search_fn = 'strong_wolfe')
           
        #self.optimizer_Adam = torch.optim.Adam(self.parameters())
        self.optimizer_Adam = torch.optim.Adam(self.parameters(),  lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_SGD = torch.optim.SGD(self.parameters(), lr=0.01)
            
        
    def forward(self, x):  
        #x = (x+1)/2
        for i in range(len(self.layers)-2):
            x = self.activation(self.linears[i](x))
        out = self.linears[-1](x)
        return out
    
    def loss(self, xt_eq, xt_lr, xt_ic):         
        res1, res2, res3 = self.residuals(xt_eq, xt_lr, xt_ic)
        return torch.sum(res1**2) + torch.sum(res2**2) + torch.sum(res3**2)
    
    def residuals(self, xt_eq, xt_lr, xt_ic):
        # Interior PDE
        xx = Variable(xt_eq, requires_grad=True)

        u = self(xx)
        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
         
        res1 = 1e-3/np.sqrt(xt_eq.shape[0])*(u_xx + self.mu*u_x + self.k*u)
       
        # LR and IC
        res2 = 1/np.sqrt(xt_lr.shape[0])*(self(xt_lr) - xt_ic)
        
        x0 = Variable(xt_lr, requires_grad=True)
        u0 = self(x0)
        u0_x = torch.autograd.grad(outputs=u0, inputs=x0, grad_outputs=torch.ones_like(u0), retain_graph=True, create_graph=True)[0] 
        
        res3 = 1e-3/np.sqrt(xt_lr.shape[0])*u0_x

        return res1, res2, res3
    
    def res_forward(self, xt_eq, xt_lr, xt_ic):
        
        xx = Variable(xt_eq, requires_grad=True)
        
        #print(xt.shape)
        u = self(xx)

        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        
        x0 = Variable(xt_lr, requires_grad=True)
        u0 = self(x0)
        u0_x = torch.autograd.grad(outputs=u0, inputs=x0, grad_outputs=torch.ones_like(u0), retain_graph=True, create_graph=True)[0] 
        
        res1 = 1e-3/np.sqrt(xt_eq.shape[0])*(u_xx + self.mu*u_x + self.k*u)
        res2 = 1/np.sqrt(xt_lr.shape[0])*self(xt_lr)
        res3 = 1e-3/np.sqrt(xt_lr.shape[0])*u0_x

        return res1, res2, res3
    
    def J_factory(self, xt_eq, xt_lr, xt_ic):
        # Detaching the parameters because we won't be calling Tensor.backward().
        params = {k: v.detach() for k, v in self.named_parameters()}
        def res_single(params, x):
            
            x = x.unsqueeze(0)
            xx = Variable(x, requires_grad=True)

            u = functional_call(self, params, xx) #SERVE LA FUCNTIONAL CALL!

            u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
            u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 

            res = u_xx + self.mu*u_x + self.k*u
            return res

        def bc_lr_single(params, x, u):
            return functional_call(self, params, x) - u #SERVE LA FUCNTIONAL CALL!
        
        def bc_prime_single(params, x):
            x = x.unsqueeze(0)
            xx = Variable(x, requires_grad=True)#
        
            u = functional_call(self, params, xx) #SERVE LA FUCNTIONAL CALL!
            u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        
            return u_x

        # Compute J(x1)
        jac1 = vmap(jacrev(res_single), (None, 0))(params, xt_eq)
        jac1 = 1e-3/np.sqrt(xt_eq.shape[0])*torch.cat([jac1[j].flatten(2) for j in jac1],dim=2)

        jac2 = vmap(jacrev(bc_lr_single), (None, 0, 0))(params, xt_lr, xt_ic)
        jac2 = 1/np.sqrt(xt_lr.shape[0])*torch.cat([jac2[j].flatten(2) for j in jac2],dim=2)
        
        jac3 = vmap(jacrev(bc_prime_single), (None, 0))(params, xt_lr)
        jac3 = 1e-3/np.sqrt(xt_lr.shape[0])*torch.cat([jac3[j].flatten(2) for j in jac3],dim=2)

        jac = torch.cat([jac1, jac2, jac3], dim=0)
        #jac = torch.cat([jac1, jac2], dim=0)
        return jac.squeeze()
        
class NN_Nonlinear(nn.Module): 
    def __init__(self, layers, activation, learning_rate = 0.001):
        super().__init__()


        self.learning_rate = learning_rate
       
        self.layers = layers #Shape of the (hidden) layers      
        self.activation = activation #Activation function
        self.loss_function = nn.MSELoss(reduction ='mean') #Loss function for singular evaluation
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data,gain =1)
            nn.init.zeros_(self.linears[i].bias.data)    
            
        self.optimizer = torch.optim.LBFGS(self.parameters(), 0.1, 
                                      max_iter = 20, 
                                      max_eval = None, 
                                      tolerance_grad = 1e-11, 
                                      tolerance_change = 1e-11, 
                                      history_size = 100, 
                                      line_search_fn = 'strong_wolfe')
           
        #self.optimizer_Adam = torch.optim.Adam(self.parameters())
        self.optimizer_Adam = torch.optim.Adam(self.parameters(),  lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_SGD = torch.optim.SGD(self.parameters(), lr=0.01)
        
    def source(self, x):
        return 2*np.pi*torch.sin(8*np.pi*x) 
            
        
    def forward(self, x):  
        #x = (x+1)/2
        for i in range(len(self.layers)-2):
            x = self.activation(self.linears[i](x))
        out = self.linears[-1](x)
        return out
    
    def loss(self, xt_eq, xt_lr, xt_ic):         
        res1, res2 = self.residuals(xt_eq, xt_lr, xt_ic)
        return torch.sum(res1**2)+torch.sum(res2**2)
    
    def residuals(self, xt_eq, xt_lr, xt_ic):
        # Interior PDE
        xx = Variable(xt_eq, requires_grad=True)

        u = self(xx)
        
        f = self.source(xt_eq)

        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
         
        res1 = 1/np.sqrt(xt_eq.shape[0])*(u*u_x + f)
       
        # LR and IC
        res2 = 1/np.sqrt(xt_lr.shape[0])*(self(xt_lr) - xt_ic)

        return res1, res2
    
    def res_forward(self, xt_eq, xt_lr, xt_ic):
        
        xx = Variable(xt_eq, requires_grad=True)
        
        #print(xt.shape)
        u = self(xx)

        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        
        res1 = 1/np.sqrt(xt_eq.shape[0])*(u*u_x)
        res2 = 1/np.sqrt(xt_lr.shape[0])*self(xt_lr)

        return res1, res2
    
    def J_factory(self, xt_eq, xt_lr, xt_ic):
        # Detaching the parameters because we won't be calling Tensor.backward().
        params = {k: v.detach() for k, v in self.named_parameters()}
        def res_single(params, x):
            
            x=x.unsqueeze(0)
            f = self.source(x)
            xx = Variable(x, requires_grad=True)

            u = functional_call(self, params, xx) #SERVE LA FUCNTIONAL CALL!

            u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 

            res = u*u_x + f
            return res

        def bc_lr_single(params, x, u):
            return functional_call(self, params, x) - u #SERVE LA FUCNTIONAL CALL!

        # Compute J(x1)
        jac1 = vmap(jacrev(res_single), (None, 0))(params, xt_eq)
        jac1 = 1/np.sqrt(xt_eq.shape[0])*torch.cat([jac1[j].flatten(2) for j in jac1],dim=2)

        jac2 = vmap(jacrev(bc_lr_single), (None, 0, 0))(params, xt_lr, xt_ic)
        jac2 = 1/np.sqrt(xt_lr.shape[0])*torch.cat([jac2[j].flatten(2) for j in jac2],dim=2)

        jac=torch.cat([jac1,jac2], dim=0)
        return jac.squeeze()
            
        
    
class NN_Nonlinear(nn.Module): 
    def __init__(self, layers, activation, learning_rate = 0.001):
        super().__init__()


        self.learning_rate = learning_rate
       
        self.layers = layers #Shape of the (hidden) layers      
        self.activation = activation #Activation function
        self.loss_function = nn.MSELoss(reduction ='mean') #Loss function for singular evaluation
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data,gain =1)
            nn.init.zeros_(self.linears[i].bias.data)    
            
        self.optimizer = torch.optim.LBFGS(self.parameters(), 0.1, 
                                      max_iter = 20, 
                                      max_eval = None, 
                                      tolerance_grad = 1e-11, 
                                      tolerance_change = 1e-11, 
                                      history_size = 100, 
                                      line_search_fn = 'strong_wolfe')
           
        #self.optimizer_Adam = torch.optim.Adam(self.parameters())
        self.optimizer_Adam = torch.optim.Adam(self.parameters(),  lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_SGD = torch.optim.SGD(self.parameters(), lr=0.01)
            
        
    def forward(self, x):  
        #x = (x+1)/2
        for i in range(len(self.layers)-2):
            x = self.activation(self.linears[i](x))
        out = self.linears[-1](x)
        return out
    
    def loss(self, xt_eq, xt_lr, xt_ic):         
        res1, res2, res3 = self.residuals(xt_eq, xt_lr, xt_ic)
        return torch.sum(res1**2) + torch.sum(res2**2) + torch.sum(res3**2)
    
    def residuals(self, xt_eq, xt_lr, xt_ic):
        # Interior PDE
        xx = Variable(xt_eq, requires_grad=True)

        u = self(xx)
        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0] 
         
        res1 = 1/np.sqrt(xt_eq.shape[0])*(u_xx + 10*torch.sin(u))
       
        # LR and IC
        res2 = 1/np.sqrt(xt_lr.shape[0])*(self(xt_lr) - xt_ic)
        
        x0 = Variable(xt_lr, requires_grad=True)
        u0 = self(x0)
        u0_x = torch.autograd.grad(outputs=u0, inputs=x0, grad_outputs=torch.ones_like(u0), retain_graph=True, create_graph=True)[0] 
        
        res3 = 1/np.sqrt(xt_lr.shape[0])*u0_x

        return res1, res2, res3
    
    def res_forward(self, xt_eq, xt_lr, xt_ic):
        
        xx = Variable(xt_eq, requires_grad=True)
        
        #print(xt.shape)
        u = self(xx)

        u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        
        x0 = Variable(xt_lr, requires_grad=True)
        u0 = self(x0)
        u0_x = torch.autograd.grad(outputs=u0, inputs=x0, grad_outputs=torch.ones_like(u0), retain_graph=True, create_graph=True)[0] 
        
        res1 = 1/np.sqrt(xt_eq.shape[0])*(u_xx + 10*torch.sin(u))
        res2 = 1/np.sqrt(xt_lr.shape[0])*self(xt_lr)
        res3 = 1/np.sqrt(xt_lr.shape[0])*u0_x

        return res1, res2, res3
    
    def J_factory(self, xt_eq, xt_lr, xt_ic):
        # Detaching the parameters because we won't be calling Tensor.backward().
        params = {k: v.detach() for k, v in self.named_parameters()}
        def res_single(params, x):
            
            x = x.unsqueeze(0)
            xx = Variable(x, requires_grad=True)

            u = functional_call(self, params, xx) #SERVE LA FUCNTIONAL CALL!

            u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
            u_xx = torch.autograd.grad(outputs=u_x, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 

            res = u_xx + 10*torch.sin(u)
            return res

        def bc_lr_single(params, x, u):
            return functional_call(self, params, x) - u #SERVE LA FUCNTIONAL CALL!
        
        def bc_prime_single(params, x):
            x = x.unsqueeze(0)
            xx = Variable(x, requires_grad=True)#
        
            u = functional_call(self, params, xx) #SERVE LA FUCNTIONAL CALL!
            u_x = torch.autograd.grad(outputs=u, inputs=xx, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] 
        
            return u_x

        # Compute J(x1)
        jac1 = vmap(jacrev(res_single), (None, 0))(params, xt_eq)
        jac1 = 1/np.sqrt(xt_eq.shape[0])*torch.cat([jac1[j].flatten(2) for j in jac1],dim=2)

        jac2 = vmap(jacrev(bc_lr_single), (None, 0, 0))(params, xt_lr, xt_ic)
        jac2 = 1/np.sqrt(xt_lr.shape[0])*torch.cat([jac2[j].flatten(2) for j in jac2],dim=2)
        
        jac3 = vmap(jacrev(bc_prime_single), (None, 0))(params, xt_lr)
        jac3 = 1/np.sqrt(xt_lr.shape[0])*torch.cat([jac3[j].flatten(2) for j in jac3],dim=2)

        jac = torch.cat([jac1, jac2, jac3], dim=0)
        #jac = torch.cat([jac1, jac2], dim=0)
        return jac.squeeze()       
    
    
    
def trainer(model, xt_eq, xt_lr, xt_ic, X_u_test, u_sol, optim = "Adam", num_epochs=20000, verbose=100):

    learning_rate = 0.001

    start_time = time.time() 
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    eta = 1
    h = 0.001
    tol = 1e-7
    alpha = 0.9999

    epoch = 0
    
    for epoch in range(num_epochs):
        
        if optim=="Adam":
            model.optimizer_Adam.zero_grad()
            loss = model.loss(xt_eq, xt_lr, xt_ic)
            loss.backward()
            model.optimizer_Adam.step()
            
        elif optim=="LM":
            
            params = {k: v.detach() for k, v in model.named_parameters()}
            J = model.J_factory(xt_eq, xt_lr, xt_ic)
            R = torch.cat(model.residuals(xt_eq, xt_lr, xt_ic), dim=0).detach()
            f_i = torch.cat(model.res_forward(xt_eq, xt_lr, xt_ic), dim=0).detach()
            I = torch.eye(J.shape[1])
            loss = 0.5*torch.sum(R**2)
            
            #S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))
            S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))

            #LM
            v=torch.linalg.lstsq(S*(J.T@J+eta*I),-J.T@R).solution
            v = (S*(v.flatten())).unsqueeze(-1)
            
            theta = torch.nn.utils.parameters_to_vector(model.parameters())
            
            
            delta = v.squeeze()/torch.linalg.norm(v.squeeze())
            theta_test = theta + h*delta
            torch.nn.utils.vector_to_parameters(theta_test, model.parameters())
            f_i_v = torch.cat(model.res_forward(xt_eq, xt_lr, xt_ic), dim=0).detach()
            f_i_vv = (2/h)*((f_i_v -f_i)/h - (J@delta).unsqueeze(1))
            
            #S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))
            S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))

            a = torch.linalg.lstsq(S*(J.T@J+eta*I),-J.T@f_i_vv).solution
            a = (S*(a.flatten())).unsqueeze(-1)

            crit = 2*torch.linalg.norm(a)/torch.linalg.norm(v)

            if crit <= alpha:
                step = v.squeeze() + 0.5*a.squeeze()
            else:
                step = v.squeeze()
                
            theta_new = theta + step
            
            torch.nn.utils.vector_to_parameters(theta_new, model.parameters())

            R_new = torch.cat(model.residuals(xt_eq, xt_lr, xt_ic),dim=0).detach()
            # Check improvement and update eta
            rho=(loss**2-(0.5*torch.sum(R_new**2))**2)/torch.abs(torch.dot(step,(eta*step.unsqueeze(-1)+J.T@R).squeeze()))
            
            count = 0

            while ((rho<tol) and (count <= 100)) :

                eta = min(eta*2, 10e8)

                torch.nn.utils.vector_to_parameters(theta, model.parameters())

                #S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))
                S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))

                #LM
                v=torch.linalg.lstsq(S*(J.T@J+eta*I),-J.T@R).solution
                v = (S*(v.flatten())).unsqueeze(-1)
                
                delta = v.squeeze()/torch.linalg.norm(v.squeeze())
                theta_test = theta + h*delta
                torch.nn.utils.vector_to_parameters(theta_test, model.parameters())
                f_i_v = torch.cat(model.res_forward(xt_eq, xt_lr, xt_ic), dim=0).detach()
                f_i_vv = (2/h)*((f_i_v -f_i)/h - (J@delta).unsqueeze(1))

                #S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))
                S = (1/torch.linalg.norm(J.T@J+eta*I, ord=torch.inf, dim=0))

                a = torch.linalg.lstsq(S*(J.T@J+eta*I),-J.T@f_i_vv).solution
                a = (S*(a.flatten())).unsqueeze(-1)

                crit = 2*torch.linalg.norm(a)/torch.linalg.norm(v)

                if crit <= alpha:
                    step = v.squeeze() + 0.5*a.squeeze()
                else:
                    step = v.squeeze()

                theta_new = theta + step

                torch.nn.utils.vector_to_parameters(theta_new, model.parameters())

                R_new = torch.cat(model.residuals(xt_eq, xt_lr, xt_ic), dim=0).detach()
                # Check improvement and update eta
                rho=(loss**2-(0.5*torch.sum(R_new**2))**2)/torch.abs(torch.dot(step,(eta*step.unsqueeze(-1)+J.T@R).squeeze()))
                count = count + 1

            eta = max(eta/7, 10e-8)

        if epoch % verbose == 0:
            
            print('Epoch ' + str(epoch) + ', Loss function (total, residuals, boundaries):')
            #loss_show = PINN.loss_showcase()
            print(loss)
            '''
            if epoch == 1000:
                learning_rate = learning_rate*0.1
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

            '''
            

    elapsed = time.time() - start_time                
    print('Training time: %.2f' % (elapsed))

    # print(results)

    ''' Model Accuracy ''' 
    xx = Variable(X_u_test, requires_grad=True)
    
    u_pred = model(xx)
    
    u_pred_x = torch.autograd.grad(outputs=u_pred, inputs=xx, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0] 
    u_pred_xx = torch.autograd.grad(outputs=u_pred_x, inputs=xx, grad_outputs=torch.ones_like(u_pred_x), retain_graph=True, create_graph=True)[0] 


    error = (u_pred.detach().cpu() - u_sol(X_u_test.cpu()))**2
    times = loss.unsqueeze(0)


    return u_pred.detach().cpu(), u_pred_x.detach().cpu(), u_pred_xx.detach().cpu(), error.detach().cpu(), times.detach().cpu()