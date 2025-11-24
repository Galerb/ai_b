import random
import numpy as np
import os
import json
import math

class NeuralNetwork:
    def __init__(self, laws: list):
        self.DEBUG = False
        self.LEARNING_RATE = 0.028
        self.val_epochs = 1000
        self.MIN_VAL = 0
        self.MAX_VAL = 169
        self.laws = laws
        self.len_print_epoch = 10
        self.type_of_activation = "ReLU"
        if len(self.laws) < 2:
            raise ValueError("Laws must be a list with at least two elements.")
        
    def __setattr__(self, name, value):
        if name not in ("list_laws", "list_z", "list_weights", "DEBUG"):
            if self.DEBUG == True:
                print(f"Setting attribute {name} to {value}")
        super().__setattr__(name, value)
        if name == "type_of_activation":
            if value == 'ReLU':
                self.init_weights('he')
            elif value in ("tanh", "sig"):
                self.init_weights("xavier")
            else:
                raise ValueError("Unsupported activation type")
        

    def init_weights(self, method="he"):
        self.list_laws = [np.zeros(i) for i in self.laws]
        self.list_z = [np.zeros(i) for i in self.laws]
        weights = []

        for i in range(len(self.laws) - 1):
            fan_in = self.laws[i]
            fan_out = self.laws[i + 1]

            if method == "he":
                w = np.random.randn(fan_out, fan_in + 1) * np.sqrt(2 / fan_in)
            elif method == "xavier":
                w = np.random.randn(fan_out, fan_in + 1) * np.sqrt(1 / ((fan_in + fan_out) / 2))
            else:
                raise ValueError("Unsupported init method")

            w[:, -1] = 0
            weights.append(w)
        self.list_weights = weights

    def save_nn(self, path, name):
        ''' принимает строки типа "C:/Users/damir/Desktop/AI" и имя файла без расширения '''
        data = {"activation" : self.type_of_activation,
                "learning_rate" : self.LEARNING_RATE, 
                "val_epochs" : self.val_epochs,
                "MIN_VAL" : self.MIN_VAL,
                "MAX_VAL" : self.MAX_VAL,
                "laws" : self.laws,
                "len_print_epoch" : self.len_print_epoch,
                "list_laws" : self.list_laws,
                "list_weights" : self.list_weights
                }
        with open(os.path.join(path, name + ".json"), "w") as save:
            json.dump(data, save, ensure_ascii=False, indent=4)
    
    def load_nn(self, path):
        pass

    def activation(self, x,  IsDerivative=False):
        ''' принимает np.array превращает весь список по переменной type_of_activation и берет производную если IsDerivative=True '''
        if self.type_of_activation == "tanh":
            if IsDerivative:
                t = np.tanh(x)
                return 1 - t ** 2
            else:
                return np.tanh(x)
        elif self.type_of_activation == "sig":
            if IsDerivative:
                s = 1 / (1 + np.exp(-x))
                return s * (1 - s)
            else:
                return 1 / (1 + np.exp(-x))
        elif self.type_of_activation == "ReLU":
            if IsDerivative:
                return np.where(x > 0, 1, 0)
            else:
                return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation function type.")

    def normalise(self, input):
        ''' принимает np.array и нормализует весь список по переменной type_of_activation '''
        if self.type_of_activation == "sig":
            return (input - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL)
        elif self.type_of_activation == 'tanh':
            return (input - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL) * 2 - 1
        elif self.type_of_activation == 'ReLU':
            return (input - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL)
        else:
            raise ValueError("Unsupported activation function type.")
        
    def denormalise(self, input):
        ''' принимает np.array и денормализует весь список по переменной type_of_activation '''
        if self.type_of_activation == "sig":
            return (input) * (self.MAX_VAL - self.MIN_VAL) + self.MIN_VAL
        elif self.type_of_activation == 'tanh':
            return  ((input + 1) / 2) * (self.MAX_VAL - self.MIN_VAL) + self.MIN_VAL
        elif self.type_of_activation == 'ReLU':
            return (input) * (self.MAX_VAL - self.MIN_VAL) + self.MIN_VAL
        else:
            raise ValueError("Unsupported activation function type.")        

    def forward(self, inp):
        ''' принимает np.array и делает прямой проход по сети '''
        self.list_laws[0] = np.array(inp)
        self.list_z[0] = np.array(inp)

        # скрытые слои с активацией
        for i in range(len(self.laws) - 2):
            temp_laws = np.append(self.list_laws[i], 1)
            z = self.list_weights[i] @ temp_laws
            self.list_z[i + 1] = z
            self.list_laws[i + 1] = self.activation(z)

        # последний слой — линейный выход
        last_idx = len(self.laws) - 2
        temp_laws = np.append(self.list_laws[last_idx], 1)
        z = self.list_weights[last_idx] @ temp_laws
        self.list_z[last_idx + 1] = z
        self.list_laws[last_idx + 1] = z

        return self.list_laws[-1]



    def backpropagation(self, inp, target):
        ''' принимает списки inp и target, делает обратное распространение ошибки и обновляет веса '''
        for i in range(len(self.list_weights)):
            for j in range(self.list_weights[i].shape[0]):
                self.list_weights[i][j] += 0
        npinput = np.asarray(inp, dtype=float)
        output = self.forward(npinput.reshape(-1))
        error = output - np.asarray(target, dtype=float).reshape(-1)
        delta = error 

        prev_acts = np.append(self.list_laws[-2].reshape(-1), 1)
        self.list_weights[-1] -= self.LEARNING_RATE * np.outer(delta, prev_acts)

        prev_delta = delta
        for i in range(len(self.list_weights) - 2, -1, -1):
            w_next_no_bias = self.list_weights[i + 1][:, :-1]

            error_i = w_next_no_bias.T @ prev_delta

            z_i1 = self.list_z[i + 1].reshape(-1)
            deriv_i = self.activation(z_i1, IsDerivative=True)

            delta_i = error_i * deriv_i

            prev_acts_i = np.append(self.list_laws[i].reshape(-1), 1)
            self.list_weights[i] -= self.LEARNING_RATE * np.outer(delta_i, prev_acts_i)

            prev_delta = delta_i

        return output

    def fit(self, X, y):
        ''' принимает списки X u y проверяет данные нормализует и обучает сеть '''
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if len(X) != len(y):
            raise ValueError("Input and target data must have the same number of samples.")
        if len(X[0]) != self.laws[0]:
            raise ValueError("Input data dimension does not match the input layer size.")
        
        indexes = np.random.permutation(len(X))
        X, y =  self.normalise(X[indexes]), self.normalise(y[indexes])

        for epoch in range(self.val_epochs):  
            for i in range(len(X)):
                self.backpropagation(X[i], y[i])
            if (epoch + 1) % self.len_print_epoch == 0:
                preds = self.predict(X)
                mae = np.mean(np.abs(preds - y))
                print(f"Epoch {epoch+1}, MAE={mae:.4f}")
            if self.len_print_epoch != 0:
                if (epoch + 1) % self.len_print_epoch == 0 or epoch == 0:
                    print(f"epoch {epoch + 1}/{self.val_epochs} completed.")

    def predict(self, X):
        ''' принимает список X и возвращает предсказания с учетом нормализации '''
        X = self.normalise(np.array(X, dtype=float))
        predictions = []
        for i in range(len(X)):
            predictions.append(self.forward(X[i]))
        return self.denormalise(np.array(predictions))

class NeuralNetworkText(NeuralNetwork):
    def __init__(self, laws_txt: list, len_of_text: int):
        self.DEBUG = False
        self.letters = self.get_common_symbols()
        self.MAX_len_of_text = len_of_text
        super().__init__([self.MAX_len_of_text] + laws_txt + [self.MAX_len_of_text])
        self.MAX_VAL = len(self.letters) - 1
        self.MIN_VAL = 0

    def get_common_symbols(self):
        ''' служебная функция со всеми допустимыми символами '''
        russian_upper = ''.join([chr(c) for c in range(ord('А'), ord('Я') + 1)]) + 'Ё'
        russian_lower = ''.join([chr(c) for c in range(ord('а'), ord('я') + 1)]) + 'ё'
        english_upper = ''.join([chr(c) for c in range(ord('A'), ord('Z') + 1)])
        english_lower = ''.join([chr(c) for c in range(ord('a'), ord('z') + 1)])
        digits = '0123456789'
        punctuation = r""" .,;:!?—–-()[]{}"'«»“”‘’…@#$%^&*_+=/<>|\~`"""

        return list(punctuation + russian_upper + russian_lower + english_upper + english_lower + digits)

    def str_to_vector(self, inp):
        ''' принимает строку и выдаёт np.array с индексами символов '''
        arr = np.zeros(self.MAX_len_of_text)
        length = min(len(inp), self.MAX_len_of_text)
        for i in range(length):
            ch = inp[i]
            if ch in self.letters:
                arr[i] = self.letters.index(ch)
            else:
                arr[i] = 0
        return arr

    def vector_to_str(self, inp):
        ''' превращает np.array с индексами в строку '''
        rinp = np.rint(inp).astype(int)
        rinp = np.clip(rinp, 0, len(self.letters) - 1)
        return ''.join(self.letters[i] for i in rinp)

    def predict_txt(self, inp):
        ''' принимает строку делает прямой проход и выдаёт строку вывода '''
        inp = self.normalise(self.str_to_vector(inp))
        return self.vector_to_str(self.denormalise(self.forward(inp)))

    def fit_txt(self, X, y):
        ''' принимает списки строк X и y и обучает нейросеть '''
        X, y = np.array(X), np.array(y)
        if len(X) != len(y):
            raise ValueError("Input and target data must have the same number of samples.")
        
        indexes = np.random.permutation(len(X))
        X, y =  X[indexes], y[indexes]
        
        for epoch in range(self.val_epochs):
            for i in range(len(X)):
                n = self.normalise(self.str_to_vector(X[i]))
                m = self.normalise(self.str_to_vector(y[i]))
                self.backpropagation(n, m)
            if (epoch + 1) % self.len_print_epoch == 0:
                    preds = self.str_to_vector(self.predict_txt(X))
                    mae = np.mean(np.abs(preds - self.str_to_vector(y)))
                    print(f"Epoch {epoch+1}, MAE={mae:.4f}")
            if self.len_print_epoch != 0:
                if (epoch + 1) % self.len_print_epoch == 0 or epoch == 0:
                    print(f"epoch {epoch + 1}/{self.val_epochs} completed.")
