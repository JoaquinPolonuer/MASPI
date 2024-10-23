import numpy as np
import os, sys
from helper_functions import cnnInitParams, loadDataset, cnnParamsToStack
from helper_functions import cnnConvolve, cnnPool
from scipy.signal import convolve as conv2

digitos = ['0','1','2','3','4','5','6','7','8','9']
dataset_folder = 'data/BaseOCR_MultiStyle'
numClasses = len(digitos)
imageDim = 28;         #% image dimension
images, labels = loadDataset(dataset_folder, digitos, numClasses, imageDim)


# Convolution Neural Network 
# ======================================================================
# Parametros
imageDim = images.shape[0] #% height/width de imagen
numFilters = 2;
filterDim = 9;
poolDim = 5;
images = images[:,:,0:11]
labels = labels[0:11]
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses)
numImages = images.shape[2] #% numbero de imagenes

# % Calcula costo y gradiente de una red neuronal convolucional simple
# % seguida de una capa densa con una funcion objetivo de tipo softmax
# %                            
# % Parametros:
# %  theta      -  unrolled vector de pesos
# %  images     -  la lista de imagenes de entrenamiento imageDim x imageDim x numImges
# %                
# %  numClasses -  numbero de clases
# %  filterDim  -  dimensiones del filtro convolucional                            
# %  numFilters -  numero de filtros
# %  poolDim    -  dimension del area de agrupamiento



# %% Obtener las matrices de parametros del roll de entrada 
# % Wc es la matriz de pesos filterDim x filterDim x numFilters
# % bc es el bias correspondiente

# % Wd es la matriz densa de la capa escondida con numClasses x hiddenSize
# % hiddenSize es el numero de unidades a la salida de la capa de pooling
# % bd es su bias correspondient
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,poolDim,numClasses)

#% reservar para los gradientes de los parametros.
Wc_grad = np.zeros((Wc.shape))
Wd_grad = np.zeros((Wd.shape))
bc_grad = np.zeros((bc.shape))
bd_grad = np.zeros((bd.shape));

# %%======================================================================
# %% PASO 1a: Forward Propagation
# %  Este paso propaga la entrada de en la capa convolucional, 
# %  subsampleo (mean pooling), y que luego son usadas como entradas de la 
# %  capa de softmax.


# %% Convolutional Layer
# %  Para cada imagen de entrada y filtro, realizar la convolucion, sumar el
# %  bias y aplicar la funcion de activacion sigmoid. Luego, subsamplear
# %  con mean pooling. Guardar los resultados de la convolucion en 
# %  activations y los resultados del pooling en activationsPooled.
# %  Es preciso conservarlos para el backpropagation.
# %  Utilizar las funciones cnnConvolve y cnnPool

convDim = imageDim-filterDim+1 #% dimension de la salida convolved
outputDim = int((convDim)/poolDim) #% dimension de la salida subsampled

#% tensor convDim x convDim x numFilters x numImages 
activations = np.zeros((convDim,convDim,numFilters,numImages))
#% tensor outputDim x outputDim x numFilters x numImages
activationsPooled = np.zeros((outputDim,outputDim,numFilters,numImages))

#%%% IMPLEMENTACION AQUI %%%
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc)
activationsPooled = cnnPool(poolDim, activations)
#########################################################
   
# % Redimensionar activations obteniendo 2D matriz, hiddenSize x numImages,
# % para Softmax layer
activationsPooled = np.reshape(activationsPooled,(-1,numImages))

# %% Softmax Layer
# %  Continuamos con la Forward propagation luego de las pooledActivations 
# %  para aplicarlas a una capa softmax standard. probs recibe los resultados
# %  numClasses x numImages que va a guardar la probabilidad de la pertenencia
# %  de cada imagen a una clase.
probs = np.zeros((numClasses,numImages))

#%%% IMPLEMENTACION AQUI %%%
forward = Wd@activationsPooled + bd

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

probs = softmax(forward)

################################################
# %%======================================================================
# %% PASO 1b: Calculo del Costo
# %  Se van a usar los labels y las probabilidades para calcular la funcion
# %  objetivo de tipo softmax. El resultado se guarda en cost.

cost = 0 #% inicializo cost

#%%% IMPLEMENTAR AQUI %%%


################################################
# %%======================================================================
# %% PASO 1c: Backpropagation
# %  Backpropagar los errores a traves de la capa de salida, convolucional y pooling.
# %  Guardar los errores para el siguiente paso para el calculo del gradiente.
# %  Backpropagar el error para la capa softmax es similar al Ejercicio 1.
# %  Para backpropagar en la capa de pooling, hay que des-agrupar el error
# %  con respecto a la capa de de pooling para cada filtro y cada imagen.
# %  Se puede usar la funcion kron y una matriz de unos para hacerlo de manera
# %  eficiente.

# %%% IMPLEMENTAR AQUI %%%



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%======================================================================
# %% PASO 1d: Calculo del Gradiente
# %  Luego de propagar los errores, deben ser usados para calcular el gradiente
# %  con respecto a los pesos. El calculo del gradiente de la capa softmax se 
# %  calculo de la manera usual. Para calcular el gradiente de un filtro en la capa convolucional
# %  se debe hacer la convolucion entren el error propagado de ese filtro 
# %  para cada imagen y acumularlo sobre todas las imagenes.

# %%% IMPLEMENTAR AQUI %%%


#############################################################################3
#%% Desenrollar los gradientes en un vector para usar luego por minFunc
gWc = np.expand_dims(Wc_grad.flatten(),1)
gWd = np.expand_dims(Wd_grad.flatten(),1)
gbc = np.expand_dims(bc_grad.flatten(),1)
gbd = np.expand_dims(bd_grad.flatten(),1)
grad = np.vstack((gWc, gWd, gbc, gbd))
