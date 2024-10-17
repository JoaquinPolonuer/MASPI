import torch
from torch import nn
import numpy as np

torch.manual_seed(0) # para tener siempre los mismos pesos iniciales

def sigmoid(x):
    s = 1 / (1 + torch.exp(-x))
    ds = s * (1 - s)
    return s, ds
    
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-1):
        # W = llegada x salida
        self.lr = lr
        
        self.Wh = torch.randn((hidden_size, input_size))         
        self.Wo = torch.randn((output_size, hidden_size))
        
        self.x = 0
        self.a_h = 0
        self.y, self.f_prime_a_h = 0, 0
        self.a_o = 0
        self.z, self.f_prime_a_o = 0, 0
        
        self.t = 0
        
        self.delta_o = 0
        self.delta_h = 0
        
    def forward(self, x):
        self.x = x.reshape(-1, 1)
        self.a_h = self.Wh @ self.x 
        self.y, self.f_prime_a_h = sigmoid(self.a_h)
        self.a_o = self.Wo @ self.y
        self.z, self.f_prime_a_o = sigmoid(self.a_o)
        return self.z
    
    def backward(self, t):
        self.t = t.reshape(-1, 1)
        self.delta_o = (self.t - self.z)*self.f_prime_a_o
        self.delta_h = self.f_prime_a_h * (self.Wo.T @ self.delta_o)
        self.Wo += self.lr*self.delta_o @ self.y.T
        self.Wh += self.lr*self.delta_h @ self.x.T
    
    def loss(self, t):
        return 0.5*torch.sum((t - self.z)**2)
        
            
        
