import numpy as np
from scipy.signal import convolve as conv2
import cv2, os


def loadDataset(dataset_folder, digitos, Nc, imageDim):

    data = []
    target = []

    for d in digitos:
        pics = os.listdir(os.path.join(dataset_folder, d))
        for pic in pics:
            try:
                i = cv2.imread(os.path.join(dataset_folder, d, pic))
            except:
                print("Problema con picture ", os.path.join(dataset_folder, d, pic))
            else:
                if len(i.shape) == 3:  # only grayscale images
                    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
                ii = cv2.resize(i, (imageDim, imageDim))
                data.append(ii)
                v = np.zeros((Nc))
                v[int(d)] = 1.0
                target.append(v)

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)

    return data.T, target.T  # se devuelve traspuesta ya que el primer axis es el nro de samples


def cnnInitParams(imageDim, filterDim, numFilters, poolDim, numClasses):
    # % Initialize parameters for a single layer convolutional neural
    # % network followed by a softmax layer.
    # %
    # % Parameters:
    # %  imageDim   -  height/width of image
    # %  filterDim  -  dimension of convolutional filter
    # %  numFilters -  number of convolutional filters
    # %  poolDim    -  dimension of pooling area
    # %  numClasses -  number of classes to predict
    # %
    # %
    # % Returns:
    # %  theta      -  unrolled parameter vector with initialized weights

    # % Initialize parameters randomly based on layer sizes.
    assert filterDim < imageDim, "filterDim must be less that imageDim"

    Wc = 1e-1 * np.random.randn(filterDim, filterDim, numFilters)

    outDim = imageDim - filterDim + 1  # % dimension of convolved image

    # % assume outDim is multiple of poolDim
    assert outDim % poolDim == 0, "poolDim must divide imageDim - filterDim + 1"

    outDim = int(outDim / poolDim)
    hiddenSize = outDim**2 * numFilters

    # % we'll choose weights uniformly from the interval [-r, r]
    r = np.sqrt(6) / np.sqrt(numClasses + hiddenSize + 1)
    Wd = np.random.rand(numClasses, hiddenSize) * 2 * r - r

    bc = np.zeros((numFilters, 1))
    bd = np.zeros((numClasses, 1))

    # % Convert weights and bias gradients to the vector form.
    # % This step will "unroll" (flatten and concatenate together) all
    # % your parameters into a vector, which can then be used with minFunc.
    theta = np.vstack(
        (
            np.expand_dims(Wc.flatten(), 1),
            np.expand_dims(Wd.flatten(), 1),
            np.expand_dims(bc.flatten(), 1),
            np.expand_dims(bd.flatten(), 1),
        )
    )

    return theta


def cnnParamsToStack(theta, imageDim, filterDim, numFilters, poolDim, numClasses):
    # % Converts unrolled parameters for a single layer convolutional neural
    # % network followed by a softmax layer into structured weight
    # % tensors/matrices and corresponding biases
    # %
    # % Parameters:
    # %  theta      -  unrolled parameter vectore
    # %  imageDim   -  height/width of image
    # %  filterDim  -  dimension of convolutional filter
    # %  numFilters -  number of convolutional filters
    # %  poolDim    -  dimension of pooling area
    # %  numClasses -  number of classes to predict
    # %
    # %
    # % Returns:
    # %  Wc      -  filterDim x filterDim x numFilters parameter matrix
    # %  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
    # %             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2
    # %  bc      -  bias for convolution layer of size numFilters x 1
    # %  bd      -  bias for dense layer of size hiddenSize x 1

    outDim = (imageDim - filterDim + 1) / poolDim
    hiddenSize = int(outDim**2 * numFilters)

    # %% Reshape theta
    indS = 0
    indE = filterDim**2 * numFilters
    Wc = np.reshape(theta[indS:indE], (filterDim, filterDim, numFilters))
    indS = indE
    indE = indE + hiddenSize * numClasses
    Wd = np.reshape(theta[indS:indE], (numClasses, hiddenSize))
    indS = indE
    indE = indE + numFilters
    bc = theta[indS:indE]
    bd = theta[indE:]

    return Wc, Wd, bc, bd


def cnnConvolve(filterDim, numFilters, images, Wc, bc):
    # %  cnnConvolve Devuelve el resultado de hacer la convolucion de W y b con
    # %  las imagenes de entrada
    # %
    # % Parametetros:
    # %  filterDim - dimension del filtro
    # %  numFilters - cantidad de filtros
    # %  images - imagenes 2D para convolucionar. Estas imagenes tienen un solo
    # %  canal (gray scaled). El array images es del tipo images(r, c, image number)
    # %  Wc, bc - Wc, bc para calcular los features
    # %         Wc tiene tamanio (filterDim,filterDim,numFilters)
    # %         bc tiene tamanio (numFilters,1)
    # %
    # % Devuelve:
    # %  convolvedFeatures - matriz de descriptores convolucionados de la forma
    # %                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    imageDim = images.shape[0]
    numImages = images.shape[2]

    convDim = imageDim - filterDim + 1

    convolvedFeatures = np.zeros((convDim, convDim, numFilters, numImages))

    # % Instrucciones:
    # %   Convolucionar cada filtro con cada imagen para obtener un array  de
    # %   tamaño (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
    # %   de modo que convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    # %   es el valor del descriptor featureNum para la imagen imageNum en la
    # %   region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
    # %
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            # % convolucion simple de una imagen con un filtro
            convolvedImage = np.zeros((convDim, convDim))
            # % Obtener el filtro (filterDim x filterDim)
            f = Wc[:, :, filterNum]

            # % Girar la matriz dada la definicion de convolucion
            # f = np.rot90(np.squeeze(f),2);
            # % Obtener la imagen
            im = np.squeeze(images[:, :, imageNum])

            # %%% IMPLEMENTACION AQUI %%%
            # % Convolucionar "filter" con "im", y adicionarlos a
            # % convolvedImage para estar seguro de realizar una convolucion
            # % 'valida'
            convolvedImageTmp = conv2(im, f, mode="valid")
            convolvedImage += convolvedImageTmp

            # %%% IMPLEMENTACION AQUI %%%
            # % Agregar el bias
            convolvedImage += bc[filterNum]

            # %%% IMPLEMENTACION AQUI %%%
            # % Luego, aplicar la funcion sigmoide para obtener la activacion de
            # % la neurona.
            convolvedFeatures[:, :, filterNum, imageNum] = 1.0 / (1 + np.exp(-convolvedImage))

    return convolvedFeatures


def cnnPool(poolDim, convolvedFeatures):
    # %  cnnPool Pools los descriptores provenientes de la convolucion
    # %  La funcion usa el Pool promedio
    # % Parametros:
    # %  poolDim - dimension de la regiom de pool
    # %  convolvedFeatures - los descriptores a realizar el pool
    # %                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    # %
    # % Devuelve:
    # %  pooledFeatures - matriz de los features agrupados
    # %                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
    # %

    numImages = convolvedFeatures.shape[3]
    numFilters = convolvedFeatures.shape[2]
    convolvedDim = convolvedFeatures.shape[0]

    px, py = int(convolvedDim / poolDim), int(convolvedDim / poolDim)
    pooledFeatures = np.zeros((px, py, numFilters, numImages))

    # % Instrucciones:
    # %   Realizar el pool de los features en regiones de tamaño poolDim x poolDim,
    # %   para obtener la matriz pooledFeatures de
    # %   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages

    # %   pooledFeatures(poolRow, poolCol, featureNum, imageNum) es el valor
    # %   del descriptor featureNum de la imagen imageNum agrupada sobre la
    # %   region (poolRow, poolCol).
    # %

    # %%% IMPLEMENTAR AQUI %%%
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            # for poolRow in range(px):
            #     for poolCol in range(py):
            #         pooledFeatures[poolRow, poolCol, filterNum, imageNum] = np.mean(
            #             convolvedFeatures[
            #                 poolDim * poolRow : poolDim * (poolRow + 1),
            #                 poolDim * poolCol : poolDim * (poolCol + 1),
            #                 filterNum,
            #                 imageNum,
            #             ]
            #         )
            features = convolvedFeatures[:,:,filterNum, imageNum]
            kernel = np.ones((poolDim, poolDim))
            temp = conv2(features, kernel, mode='valid')
            pooled = temp[::poolDim,::poolDim]
            pooledFeatures[:,:, filterNum, imageNum] = pooled
            
    return pooledFeatures
