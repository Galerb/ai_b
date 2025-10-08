import random
import math
import numpy as np

class NeuralNetwork:
    def __init__(self, laws = None):
        self.type_of_activation = "tanh"
        self.LEARNING_RATE = 0.028
        self.val_epochs = 1000
        self.MIN_VAL = 0
        self.MAX_VAL = 169
        self.laws = laws
        self.len_print_epoch = 10
        self.list_laws = [
            [0 for _ in range(self.laws[i])]
            for i in range(1, len(self.laws) - 1)
        ]
        self.list_weights = [
            [
                [random.uniform(0.01, 1) for _ in range(self.laws[i] + 1)]
                for _ in range(self.laws[i + 1])
            ]
            for i in range(len(self.laws) - 1)
        ]

    def activation(self, x,  IsDerivative=False):
        if self.type_of_activation == "tanh":
            if IsDerivative:
                return 1 - x ** 2 
            else:
                return math.tanh(x)
        elif self.type_of_activation == "sig":
            if IsDerivative:
                return x * (1 - x)
            else:
                return 1 / (1 + math.exp(-x))
        elif self.type_of_activation == "ReLU":
            if IsDerivative:
                return 1 if x > 0 else 0
            else:
                return max(0.0001, x)

    def normalise(self, input):
        if type(input) == list:
            response_list = []
            for i in range(len(input)):
                response_list.append(self.normalise(input[i]))
            return response_list
        else:
            if self.type_of_activation == "sig":
                return (input - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL)
            elif self.type_of_activation == 'tanh':
                return (input - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL) * 2 - 1
            elif self.type_of_activation == 'ReLU':
                return (input - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL)
        
    def denormalise(self, input):
        if type(input) == list:
            response_list = []
            for i in range(len(input)):
                response_list.append(self.denormalise(input[i]))
            return response_list
        else:
            if self.type_of_activation == "sig":
                return (input) * (self.MAX_VAL - self.MIN_VAL) + self.MIN_VAL
            elif self.type_of_activation == 'tanh':
                return  ((input + 1) / 2) * (self.MAX_VAL - self.MIN_VAL) + self.MIN_VAL
            elif self.type_of_activation == 'ReLU':
                return (input) * (self.MAX_VAL - self.MIN_VAL) + self.MIN_VAL

    def forward(self, inp):
        current_input = inp[:]

        for layer_idx in range(len(self.list_weights)):
            next_input = []

            for neuron_weights in self.list_weights[layer_idx]:
                activation_input = 0

                for j in range(len(current_input)):
                    activation_input += current_input[j] * neuron_weights[j]

                activation_input += neuron_weights[-1]  
                next_input.append(self.activation(activation_input))

            if layer_idx < len(self.list_laws):
                self.list_laws[layer_idx] = next_input

            current_input = next_input

        return current_input

    def backpropagation(self, inp, target):

        output = self.forward(inp)
        errors = [output[i] - target[i] for i in range(len(output))]
        deltas = [None for _ in range(len(self.list_weights))]
        last_hidden = self.list_laws[-1] if self.list_laws else inp
        deltas[-1] = []
        for i in range(len(output)):
            d = errors[i] * self.activation(output[i], IsDerivative=True)
            deltas[-1].append(d)

        # === ДЕЛЬТЫ СКРЫТЫХ СЛОЁВ (в обратном порядке) ===
        for layer_idx in range(len(self.list_laws) - 1, -1, -1):
            current_activations = self.list_laws[layer_idx]
            next_deltas = deltas[layer_idx + 1]
            next_weights = self.list_weights[layer_idx + 1]

            deltas[layer_idx] = []
            for i in range(len(current_activations)):
                err = 0
                for j in range(len(next_deltas)):
                    # Без bias веса
                    err += next_deltas[j] * next_weights[j][i]
                delta = err * self.activation(current_activations[i], IsDerivative=True)
                deltas[layer_idx].append(delta)

        # === ОБНОВЛЕНИЕ ВЕСОВ ===
        for layer_idx in range(len(self.list_weights)):
            inputs = inp if layer_idx == 0 else self.list_laws[layer_idx - 1]

            for neuron_idx in range(len(self.list_weights[layer_idx])):
                for input_idx in range(len(inputs)):
                    self.list_weights[layer_idx][neuron_idx][input_idx] -= (
                        deltas[layer_idx][neuron_idx] * inputs[input_idx] * self.LEARNING_RATE
                    )
                # bias
                self.list_weights[layer_idx][neuron_idx][-1] -= (
                    deltas[layer_idx][neuron_idx] * 1 * self.LEARNING_RATE
                )

    def shuffle_data(self, X, y):
        combined = list(zip(X, y))
        random.shuffle(combined)
        X_shuffled, y_shuffled = zip(*combined)
        return list(X_shuffled), list(y_shuffled)

    def fit(self, X, y):
        for epoch in range(self.val_epochs):  
            X_shuffled, y_shuffled = self.shuffle_data(X, y)
            for i in range(len(X)):
                inp = self.normalise(X_shuffled[i])  
                target = self.normalise(y_shuffled[i])
                self.backpropagation(inp, target)
            if self.len_print_epoch != 0:
                if (epoch + 1) % self.len_print_epoch == 0 or epoch == 0:
                    print(f"epoch {epoch + 1}/{self.val_epochs} completed.")

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            pred = self.forward(self.normalise(X[i]))
            predictions.append(self.denormalise(pred))
        return predictions

    def score(self, y_pred, y_real, Type="MAE"):
        respond_list = []
        for i in range(len(y_pred)):
            pred = y_pred[i][0]
            real = y_real[i][0]
            if Type == "MSE":
                respond_list.append((pred - real) ** 2)
            elif Type == "MAE":
                respond_list.append(abs(pred - real))
        return sum(respond_list) / len(respond_list)
    


class NeuralNetworkText:
    def __init__(self):
        self.letters = self.get_common_symbols()
        self.MAX_len_of_text = 5

    def get_common_symbols(self):
        russian_upper = ''.join([chr(c) for c in range(ord('А'), ord('Я') + 1)]) + 'Ё'
        russian_lower = ''.join([chr(c) for c in range(ord('а'), ord('я') + 1)]) + 'ё'
        english_upper = ''.join([chr(c) for c in range(ord('A'), ord('Z') + 1)])
        english_lower = ''.join([chr(c) for c in range(ord('a'), ord('z') + 1)])
        digits = '0123456789'
        punctuation = r""" .,;:!?—–-()[]{}"'«»“”‘’…@#$%^&*_+=/<>|\~`"""

        return list(punctuation + russian_upper + russian_lower + english_upper + english_lower + digits)

    def fill_to_length(self, inp):
        while len(inp) < self.MAX_len_of_text:
            inp.append(0)
        return inp

    def str_to_vector(self, inp):
        response = []
        for i in range(len(inp)):
            response.append(self.letters.index(inp[i]))
        return self.fill_to_length(response)

    def vector_to_str(self, inp):
        response = []
        for i in range(len(inp)):
            response.append(self.letters[round(inp[i])])
        return response


#ent = list(map(float, input().split()))
'''ent = input()
nnt = NeuralNetworkText()
print(*nnt.vector_to_str(nnt.str_to_vector(ent)))'''