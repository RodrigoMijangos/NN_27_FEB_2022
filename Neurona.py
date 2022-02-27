import numpy as np


class Neurona:

    # Iteración actual
    k = 0

    # DataSet
    X = np.ones([3, 3])

    # Pesos generados
    W = np.ones(3)

    # Sumatoria de los pesos y X
    U = np.ones(3)

    # Y Deseada
    YD = np.ones(3)

    # Y creada a partir de valores mayores a 0 y menores o iguales a 0
    YC = np.ones(3)

    # Vector de Error
    E = np.ones(3)

    # Tasa de aprendizaje
    eta = 1

    # Registro de lo que pasa
    log = []

    def __init__(self, **parametros):
        # Transforma lo que venga en X a una matriz o vector según corresponda
        self.X = np.array(parametros.get('x'))

        # Transforma lo que venga en W a una matriz o vector según corresponda
        self.W = np.array(parametros.get('w'))

        # Transforma lo que venga en YD a una matriz o vector según corresponda
        self.YD = np.array(parametros.get('yd'))

        self.eta = parametros.get('eta')

    def iniciarNeurona(self):

        terminado = False

        while not terminado:
            self.U = self.__calcularU()
            self.YC = self.__funcionActivacion()
            self.E = self.__calcularError()
            self.W = self.__calcularW()
            if np.array_equiv(self.YD,self.YC):
                terminado = True

        return terminado

    # Función de Activación de la Neurona
    def __funcionActivacion(self):
        YC = []

        # For que extrae los valores contenidos en U representados en value
        for value in self.U:

            # Si el valor es menor o igual a 0 entonces agrega 0 a Y creada
            if value <= 0:
                YC.append(0)

            # Si el valor es mayor a 0 entonces agrega 1 a Y creada
            if value > 0:
                YC.append(1)

        # Regresa un vector de numpy
        return np.array(YC)

    def __calcularU(self):
        # Calcula U multiplicando X por W Transpuesto
        mul = self.X * self.W

        # Suma los coeficientes de las filas
        U = np.sum(mul, axis=1)

        # Devuelve U
        return U

    def __calcularError(self):
        # Devuelve el resultado de la resta entre Y deseada y Y creada
        return self.YD - self.YC

    def __calcularW(self):
        # Crea una matriz de X columnas por 1 fila
        et = np.array([self.E])

        # Multiplica E transpuesta por X
        etx = et.T * self.X

        # Suma las filas del resultado de la matriz y lo multiplica por ETA
        etx_eta = self.eta * (np.sum(etx, axis=0))

        # Suma el vector resultante con W
        new_w = self.W + etx_eta

        # Eleva el vector al cuadrado
        tasa_e = np.multiply(self.E, self.E)

        # Suma los coeficientes del vector
        tasa_e = np.sum(tasa_e)

        # Saca la raíz cuadrada de la suma
        tasa_e = np.sqrt(tasa_e)

        # Crea un nuevo registro
        reg = {
            'K': self.k,
            'W': self.W,
            'U': self.U,
            'YC': self.YC,
            'E': self.E,
            'nW': new_w,
            'error': tasa_e
        }

        # Añade el registro al LOG
        self.log.append(reg)

        # Aumenta uno a K
        self.k += 1

        # Devuelve el nuevo valor de w
        return new_w

    def __str__(self):

        str = ''

        if len(self.log) == 0:
            return 'No hay registros aún\n'

        for log in self.log:
            for k in log.keys():
                str += f'"{k}": {log[k]}\n'
            str += f'_____________________________________________\n'

        return str
