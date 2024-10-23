import numpy as np
import os, sys
from helper_functions import cnnConvolve, loadDataset
# ======================================================================
#  PASO 1: Implementar y Testear la convolucion
#   En esta etapa, se debe implementar la operacion de convolucion. Se
#   ofrece un test en un pequeño set para asegurar que la implementacion es
#   correcta.
filterDim = 8          #% tamanio del filtro
numFilters = 100         #% cantidad de filtros
poolDim = 3          #% dimension of pooling region

#  1a: Implementar la funcion cnnConvolve en helper_functions.py
#  Usamos un set pequeño de imagenes: las 8 primeras
digitos = ['0','1','2','3','4','5','6','7','8','9']
dataset_folder = 'data/BaseOCR_MultiStyle'
Nc = len(digitos)
imageDim = 28;         #% image dimension
images, labels = loadDataset(dataset_folder, digitos, Nc, imageDim)
convImages = images[:, :, 0:8]; 


# Inicializo matrices aleatorias de pesos que van a ser utilizadas en la
# verificacion de la funcion de Convolucion
W = np.random.randn(filterDim,filterDim,numFilters);
b = np.random.rand(numFilters);
# Ahora realizamos la operacion de convolucion de las imagenes llamando a
# cnnConvolve y pasandole todos los parametros reque
convolvedFeatures = cnnConvolve(filterDim, numFilters, convImages, W, b);

#  1b: Chequeamos la convolucion encontrada
#   El codigo siguiente se chequea el resultado devuelto por cnnConvolve.
#   To ensure that you have convolved the features correctly, we have
#   provided some code to compare the results of your convolution with
#   activations from the sparse autoencoder

#   Tomando 1000 puntos random
for i in range(1000):   
    filterNum = np.random.randint(0, numFilters)
    imageNum = np.random.randint(0, 8);
    imageRow = np.random.randint(0, imageDim - filterDim + 1);
    imageCol = np.random.randint(0, imageDim - filterDim + 1);    
   
    patch = convImages[imageRow:imageRow + filterDim, imageCol:imageCol + filterDim, imageNum]
    w = np.rot90(W[:,:,filterNum],2)
    feature = np.sum(patch * w )+b[filterNum];
    feature = 1. / ( 1. + np.exp(-feature)) # sigmoide
    
    if np.abs(feature - convolvedFeatures[imageRow, imageCol,filterNum, imageNum]) > 1e-4:
        print('La salida de cnnConvolve no coincide con el test\n');
        print('Numero de filtro  : %d\n' % filterNum);
        print('Numero de imagen  : %d\n' % imageNum);
        print('Fila Imagen       : %d\n' % imageRow);
        print('Columna Imagen    : %d\n' % imageCol);
        print('Convolved feature : %0.5f\n' % convolvedFeatures[imageRow, imageCol, filterNum, imageNum]);
        print('Test feature : %0.5f\n' % feature);       
        sys.exit('La salida de cnnConvolve no coincide con el test')

print('Felicitacines! Su implementacion paso el test.')

