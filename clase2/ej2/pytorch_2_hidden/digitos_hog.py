import os
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
from torch import optim
import torch
from model import MLP

torch.manual_seed(0) # para tener siempre los mismos pesos iniciales

digitos = ['0','1','2','3','4','5','6','7','8','9']

dataset_folder = "data/BaseOCR_MultiStyle"

# PARCHE PARAMETROS
W_PARCHE = 12
H_PARCHE = 24
## HOG PARAMETROS
HOG_PIX_CELL        = 4
HOG_CELL_BLOCK      = 2
HOG_ORIENTATIONS    = 8
HOG_FEATURE_LENGTH  = 320



Nc = len(digitos)    
Nh = 18

data = []
target = []

for d in digitos:
    pics = os.listdir(os.path.join(dataset_folder,d))
    for pic in pics:
        i = cv2.imread(os.path.join(dataset_folder,d,pic))
        if len(i.shape) == 3: # only grayscale images
            i = cv2.cvtColor(i,cv2.COLOR_RGB2GRAY)
        ii = cv2.resize(i, (W_PARCHE,H_PARCHE))
        fd = hog(ii, orientations=HOG_ORIENTATIONS, 
                    pixels_per_cell=(HOG_PIX_CELL, HOG_PIX_CELL), 
                    cells_per_block=(HOG_CELL_BLOCK, HOG_CELL_BLOCK))
        data.append(fd)
        v = np.zeros((Nc))
        v[int(d)] = 1.
        target.append(v)
            
data = np.array(data, dtype=np.float32)
target = np.array(target, dtype=np.float32)

# random split for train and test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
        
# declaracion modelo
model = MLP(HOG_FEATURE_LENGTH, Nh, Nc)
loss_fnc = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

nepochs = 100

for epoch in range(nepochs):
    running_loss = 0.
    running_correct = 0.
    model.train()
    for i, [x,t] in enumerate(zip(X_train, y_train)):
        x = torch.tensor(x)
        t = torch.tensor(t)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fnc(output, t)
        loss.backward()
        optimizer.step()
        # estadisticos
        running_loss += loss.item()
        _,preds = torch.max(output,0)
        _,labels = torch.max(t,0)
        running_correct += torch.sum(preds == labels)
        
    epoch_loss = running_loss / (i-1)
    epoch_correct = running_correct / (i-1)

    running_loss = 0.
    running_correct = 0.
    model.eval()
    for j, [x,t] in enumerate(zip(X_test, y_test)):
        x = torch.tensor(x)
        t = torch.tensor(t)
        with torch.no_grad():
            output = model(x)
            # estadisticos
            _,preds = torch.max(output,0)
            _,labels = torch.max(t,0)
            running_correct += torch.sum(preds == labels)
    
    val_correct = running_correct / (j-1)    
    print('Epoca: %d - train loss: %f - train correctos: %f - test correctos: %f' % \
          (epoch, epoch_loss, epoch_correct,val_correct))
    
        