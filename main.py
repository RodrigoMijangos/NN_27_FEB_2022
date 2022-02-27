from Neurona import Neurona
import numpy as np

X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

W = np.array([0.854, 0.327, 0.558])

Y = np.array([0, 0, 0, 1])

n = Neurona(x=X, w=W, yd=Y, eta=0.4)

if n.iniciarNeurona():
    print(n)