# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:28 2021

@author: pnegr
"""

import numpy as np

## La funcion de activacion es tanh. Se devuelve tambien la derivada.
def activation(x):
    b = 2.5;
    x = b*x
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=(1-t**2)
    return t,b*dt

# La entrada consiste en un array cuyas columnas son los ejemplos de
# aprendizaje y cuyas lineas son los descriptores
X = np.array([[1, 1, -1, -1],[ 1, -1, 1, -1]])
[Ni, M]          = X.shape
No		         = 1
Nh               = 2 
Theta            = 10e-3
eta              = 0.25 # ratio de aprendizaje
epochs           = 10 # son las iteraciones para el aprendizaje
# Las salidas esperadas o targets
T = np.array([-1, 1, 1, -1])

# Se inicializa la red
# Hay una sola neurona de salida, con lo cual entre las neuronas escondidas
# y la salida hay un vector de pesos. Entre la entrada y las neuronas
# escondidas hay una matriz de pesos.

# Hidden weights (NhxNi)
Wh = np.array([[0.3615, -1.4145], [ -0.8916,  0.2010]]).reshape((Nh,Ni))
# Output weights (NoxNh)
Wo = np.array([-1.1678 ,-0.2166]).reshape((1,Nh))
# Los bias se fijan a un valor igual a 1. No hay aprendizaje en ellos.
bo = 1
bh = 1

J       = np.zeros((epochs))
J[0]    = 1e3
m       = 0
while m < epochs:
    for i in range(M):
        # voy a tomar uno a uno los puntos para actualizar los pesos
        Xm = X[:,i].reshape((Ni,1)) 
        tm = T[i]
        # Forward propagation desde la entrada X
        # Calcular primero las aj en las neuronas escondidas
        a_h				=  Wh@Xm + bh
        [y, dfh]		= activation(a_h)
        # Calcular ahora el valor de salida utilizando lo precedente
        a_o				=  Wo@y + bo
        [zm, dfo]	= activation(a_o);
        # Ya se puede calcular la salida y el error cometido
        # Evaluar ahora el delta_o a la salida: 
        delta_o		=  (tm-zm)*dfo
        #...y delta_h: 
        delta_h		= dfh*Wo.T@delta_o
        #% Ahora se actualizan los pesos
        ## los pesos de la capa de salida
        Wo				=  Wo + eta*delta_o@y.T
        ##% los pesos de la capa escondida
        Wh				=  Wh + eta*delta_h@Xm.T
            
    #Calculate total error
    J[m]    = 0;
    for i in range(M): 
        Xm = X[:,i].reshape((Ni,1))
        a_h = Wh@Xm + bh
        [y, dfh]		= activation(a_h)
        a_o = Wo@y + bo
        [zm, dfo]	= activation(a_o);
        J[m] = J[m] + (T[i] - zm)**2;
    
    J[m] = J[m]/M; 
    print('Iteracion %d: Error Total %f' % (m, J[m]))
    m = m + 1;

# El error a la salida debe ser aproximadamente 0.00071965
expErr = 0.00071965;
assert(np.abs(J[epochs-1]-expErr) < 1e-6, 'Error de implementacion')

# Resultado
for i in range(M):
    Xm = X[:,i].reshape((Ni,1))
    [y, dfh]		= activation(Wh @ Xm + bh*np.ones((Ni,1)))
    [zm, dfo]	= activation(Wo @ y + bo);
    
    print('X: [%d,%d], -> T(%d) Esperado: %f - Calculado: %f' % (X[0,i], X[1,i], i, T[i], zm))
