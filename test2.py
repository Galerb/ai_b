from ai_based import NeuralNetwork
import random
import math

def fill_to_length(inp, target_len):
    while len(inp) < target_len:
        inp.insert(0, 0)
    return inp

def is_composite(n):
    if n < 2:
        return True
    for j in range(2, int(n*0.5)):
        if n % j == 0:
            return True
    return False

X = []
y = []

for number in range(1, 100):
    digits = list(str(number))
    padded_digits = fill_to_length(digits, 5)

    padded_digits_int = []
    for digit in padded_digits:
        padded_digits_int.append(int(digit))

    if is_composite(number):
        label = [100]
    else:
        label = [0]

    X.append(padded_digits_int)
    y.append(label)

nn = NeuralNetwork([5, 20, 20, 20, 1])
nn.val_epochs = 500
nn.LEARNING_RATE = 0.05
nn.MIN_VAL = 0
nn.MAX_VAL = 100
nn.type_of_activation = "ReLU"

nn.fit(X, y)

preds = nn.predict(X)
mae = nn.score(preds, y)
print("MAE:", mae)

print(X[:10], y[:10], preds[:10], sep="\n")