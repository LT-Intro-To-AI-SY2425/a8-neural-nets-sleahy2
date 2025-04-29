from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")



print("\n<<<<<<<<<<<<<< IRIS >>>>>>>>>>>>>>\n")

import random

def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != 5:
        return None
    inputs = list(map(float, parts[:4]))
    label = parts[4]
    if label == "Iris-setosa":
        output = [1, 0, 0]
    elif label == "Iris-versicolor":
        output = [0, 1, 0]
    else:
        output = [0, 0, 1]
    return inputs, output

def normalize(data):
    num_features = len(data[0][0])
    mins = [float("inf")] * num_features
    maxs = [float("-inf")] * num_features
    for inputs, _ in data:
        for i in range(num_features):
            mins[i] = min(mins[i], inputs[i])
            maxs[i] = max(maxs[i], inputs[i])
    for inputs, _ in data:
        for i in range(num_features):
            if maxs[i] != mins[i]:
                inputs[i] = (inputs[i] - mins[i]) / (maxs[i] - mins[i])

iris_data = []
with open("iris.data") as f:
    for line in f:
        parsed = parse_line(line)
        if parsed:
            iris_data.append(parsed)

normalize(iris_data)
random.shuffle(iris_data)
split = int(0.8 * len(iris_data))
train_data = iris_data[:split]
test_data = iris_data[split:]

iris_net = NeuralNet(4, 5, 3)
iris_net.train(train_data, print_interval=200)

# Accuracy check
correct = 0
for x, y in test_data:
    out = iris_net.evaluate(x)
    if out.index(max(out)) == y.index(max(y)):
        correct += 1

accuracy = correct / len(test_data)
print(f"Test Accuracy: {accuracy:.2f}")
