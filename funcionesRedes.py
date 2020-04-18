import numpy as np
from functools import reduce


#funcion para aplanar matrices 
flatten_list_of_arrays = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

#funcion para pasae las entradas de las capas a las salidas 
sigmoid = lambda z: 1.0/(1.0+np.exp(-z))


#funcion para convertir un array plano de thetas en matrices
def inflate_matrixes(flat_thetas, shapes):
    layers = len(shapes)+1
    sizes = [shape[0]*shape[1] for shape in shapes]
    steps = np.zeros(layers, dtype=int)
    
    for i in range(layers-1):
        steps[i+1]=steps[i]+sizes[i]
        
    return[
        flat_thetas[steps[i]: steps[i+1]].reshape(*shapes[i])
        for i in range(layers-1)
    ]


#funcion para declarar las deltas 
def deltaDeclaration(shape):
    delta = []
    for sh in shape:
        temp = np.zeros(sh, dtype=int)
        delta.append(temp)
    return delta


def feed_forward(thetas, X):
    a = [X]
    for i in range(len(thetas)):
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((np.ones(len(X)).reshape(len(X),1),
                    a[i]
                	)), thetas[i].T
                )
            )
        )
    return a


def back_propagation(activate,lay,result,theta):
    deltas = [*range(lay-1), activate[-1]-result]
    for i in range(lay-2,0,-1):
        deltas[i] = np.matmul(deltas[i+1],(np.delete(theta[i],0,1)))*(activate[i]*(1-activate[i]))
    return deltas


#funcion de costo
def cost_function(flat_thetas, shapes, X, Y):
    act_array = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )

    return -(Y * np.log(act_array[-1]) + (1 - Y) * np.log(1 - act_array[-1])).sum() / len(X)

#funcion que hace el calculo de la prediccion
def prediction_calculation(flat_thetas, shapes, X, Y):
        Deltas = []
        m, layers = len(X), len(shapes)+1
        thetas = inflate_matrixes(flat_thetas,shapes)
        act_array = feed_forward(thetas,X)

        
        deltas = [*range(layers-1), act_array[-1]-Y]
        for i in range(layers-2,0,-1):
            deltas[i] = np.matmul(deltas[i+1],(np.delete(thetas[i],0,1)))*(act_array[i]*(1-act_array[i]))
            
        for i in range(layers-1):
            Deltas.append(
                (
                    np.matmul(
                        deltas[i+1].T,
                        np.hstack((
                            np.ones(len(act_array[i])).reshape(len(act_array[i]),1),
                            act_array[i])))
                )/m
            )
        Deltas = np.asarray(Deltas)
        
        return flatten_list_of_arrays(Deltas)




