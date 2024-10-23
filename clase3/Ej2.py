import numpy as np
import os, sys
from helper_functions import cnnPool
# %%======================================================================
# %% Ej 2: Implementacion y Test de Average Pooling
# %  Aqui implementamos la funcion cnnPool en helpers_functions.py
#%% 2b: Chequeo de la implementacion
testMatrix = np.arange(0,64)
testMatrix = testMatrix.reshape((8, 8))

expectedMatrix = np.array([[np.mean(np.mean(testMatrix[0:4, 0:4])), np.mean(np.mean(testMatrix[0:4, 4:8]))],
                  [np.mean(np.mean(testMatrix[4:8, 0:4])), np.mean(np.mean(testMatrix[4:8, 4:8]))]])
            
testMatrix = np.reshape(testMatrix, (8, 8, 1, 1))
        
pooledFeatures = np.squeeze(cnnPool(4, testMatrix))

if not (pooledFeatures.all() == expectedMatrix.all()):
    print('Pooling incorrecto')
    print('Esperado')
    print(expectedMatrix)
    print('Resultado')
    print(pooledFeatures)
    print('Pooling incorrecto')
else:
    print('Felicitaciones! El codigo de pooling es correcto.')


