import torch
from torch import nn

torch.manual_seed(0) # para tener siempre los mismos pesos iniciales

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
        # Inicializacion de pesos capa 1
        nn.init.xavier_uniform_(self.layer1.weight)
        # pongo a cero los bias
        self.layer1.bias.data.fill_(0)
        # Inicializacion de pesos capa 2
        nn.init.xavier_uniform_(self.layer2.weight)
        # pongo a cero los bias
        self.layer2.bias.data.fill_(0)


    def forward(self, input):
        out = self.layer1(input)
        out = torch.sigmoid(out)
        out = self.layer2(out)
        return torch.sigmoid(out)
