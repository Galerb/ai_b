from NeuralNetwork import NeuralNetwork

X = []
y = []

for i in range(1, 50):
    for j in range(1, 50):
        X.append([i, j])
        y.append([j, i])

nn = NeuralNetwork([2, 3, 4, 1])
nn.val_epochs = 100
nn.type_of_activation = "ReLU"

nn.fit(X, y)

preds = nn.predict(X)
print("MAE:", nn.score(preds, y))
print("")